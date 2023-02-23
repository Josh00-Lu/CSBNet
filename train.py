import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import net as  net
from function import normal, calc_mean_std
from sampler import InfiniteSamplerWrapper

device_ids=[]

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)  
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_network(vgg, KC, KS):
    vgg = nn.Sequential(*list(vgg.children())[:44])
    with torch.no_grad():
        network = net.Net(vgg, KC=KC, KS=KS)
    network.train()
    network.to(device)
    network = nn.DataParallel(network, device_ids=device_ids)
    return network

def load_dataset(content_dir, style_dir):
    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(content_dir, content_tf)
    style_dataset = FlatFolderDataset(style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    return content_iter, style_iter

def save_pth(network, i):
    global save_dir
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter or i == 1:
        torch.save(network.module.csbnet.state_dict(), '{:s}/csbnet_iter_{:d}.pth'.format(save_dir, i+1))   

def calc_content_loss(input, target):
    assert (input.size() == target.size())
    return nn.MSELoss()(input, target)

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return nn.MSELoss()(input_mean, target_mean) + nn.MSELoss()(input_std, target_std)
            
def get_total_loss(network, content, style, args):
    Ics = network(content, style)
    content_feats = network.module.encode_with_intermediate(content)
    style_feats = network.module.encode_with_intermediate(style)
    Ics_feats = network.module.encode_with_intermediate(Ics)
    F_c_enhanced_feats = network.module.csbnet.crsp_c(content_feats[-2])
    F_s_enhanced_feats = network.module.csbnet.crsp_s(style_feats[-2])
    
    ####################### Perceptual Loss #######################
    # content loss
    loss_content = calc_content_loss(
        Ics_feats[-1], content_feats[-1]
    ) + calc_content_loss(
        Ics_feats[-2],content_feats[-2]
    )
    # style loss
    loss_style = calc_style_loss(Ics_feats[0], style_feats[0])
    for i in range(1, 5):
        loss_style += calc_style_loss(Ics_feats[i], style_feats[i])
        
    ################# Component Enhancement Loss ###############
    loss_c_component = calc_content_loss(Ics_feats[-2], F_c_enhanced_feats)
    loss_s_component = calc_style_loss(Ics_feats[-2], F_s_enhanced_feats)
    
    ################# Smooth Loss ##############################
    # total variation loss
    loss_tv = torch.sum(torch.abs(Ics[:, :, :, :-1] - Ics[:, :, :, 1:])) + torch.sum(torch.abs(Ics[:, :, :-1, :] - Ics[:, :, 1:, :]))

    # illumination loss
    s = torch.empty(1)
    t  = torch.empty(content.size())
    std = torch.nn.init.uniform_(s, a=0.01, b=0.02)
    noise = torch.nn.init.normal(t, mean=0, std=std[0]).cuda()
    content_noise = content + noise
    Ics_N = network(content_noise, style)
    
    loss_illum = calc_content_loss(Ics_N, Ics)
  
    ################# Training Loss ############################
    L_percep = args.lambda_content * loss_content + args.lambda_style * loss_style
    L_comp = args.lambda_c_comp * loss_c_component + args.lambda_s_comp * loss_s_component
    L_smooth = args.lambda_tv * loss_tv + args.lambda_illum * loss_illum
    
    return L_percep + L_comp + L_smooth

def train(content_images, style_images, network, optimizer, i, args):
    loss = get_total_loss(network, content_images, style_images, args)
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()
    save_pth(network, i)
    return loss

def create_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--KC', type=int, default=4)
    parser.add_argument('--KS', type=int, default=-10)
    
    parser.add_argument('--lambda_content', type=float, default=3.0)
    parser.add_argument('--lambda_style', type=float, default=10.0)
    parser.add_argument('--lambda_c_comp', type=float, default=3)
    parser.add_argument('--lambda_s_comp', type=float, default=1)
    parser.add_argument('--lambda_tv', type=float, default=1e-5)
    parser.add_argument('--lambda_illum', type=float, default=3000)
    
    parser.add_argument('--content_dir', default='../datasets/MSCOCO/train2017', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', default='../datasets/wikiarts/train', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg_path', type=str, default='./models/vgg_normalised.pth')
    parser.add_argument('--save_base', type=str, default='.')
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=320000)
    parser.add_argument('--batch_size', type=int, default=4)
    
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--use_cuda', type=int, default=1)
    parser.add_argument('--gpu_num', type=int, default=1)

    args = parser.parse_args()
    
    global save_dir
    save_dir = args.save_base + '/experiments_KC='+str(args.KC)+'_KS='+str(args.KS)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return args

if __name__ == '__main__':
    cudnn.benchmark = True
    Image.MAX_IMAGE_PIXELS = None  
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    args = create_parser_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(args.gpu_num):
        device_ids.append(i)

    vgg = net.vgg
    vgg.load_state_dict(torch.load(args.vgg_path))
    network = create_network(vgg, args.KC, args.KS)

    content_iter, style_iter = load_dataset(args.content_dir, args.style_dir)
   
    optimizer = torch.optim.Adam(network.module.csbnet.parameters(), lr=args.lr)
  
    pbar = tqdm(range(args.max_iter))
    for i in pbar:
        adjust_learning_rate(optimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        loss = train(content_images, style_images, network, optimizer, i, args)
        pbar.set_description(f'loss: {loss:.4f}')

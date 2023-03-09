import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename, splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import net as net
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read image files
def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        if x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png'):
            paths.append(os.path.join(img_dir, x))
    # return [os.path.join(img_dir,x) for x in files]
    return paths

#loading the images
def load_images(content_dir, style_dir):
    if os.path.isdir(content_dir):
        content_paths = get_files(content_dir)
    else:  # Single image file
        content_paths = [content_dir]
    if os.path.isdir(style_dir):
        style_paths = get_files(style_dir)
    else:  # Single image file
        style_paths = [style_dir]
    return content_paths, style_paths

def test_transform(size, crop):
    transform_list = []
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform(ori_size):
    transform_list = []
    max_s = int (np.max(ori_size))
    thresh = 512
    if max_s > thresh:
        ratio = max_s / thresh 
        current_size = (int(ori_size[0] / ratio), int(ori_size[1] / ratio))
        transform_list.append(transforms.Resize(current_size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform(ori_size):
    resized = 0
    transform_list = []
    max_s = int (np.max(ori_size))
    thresh = 512
    if max_s > thresh:
        resized = 1
        ratio = max_s / thresh
        current_size = (int(ori_size[0] / ratio), int(ori_size[1] / ratio))
        transform_list.append(transforms.Resize(current_size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform, resized

def resize_transform(ori_size):
    transform_list = [
        transforms.Resize(ori_size)
    ]
    transform = transforms.Compose(transform_list)
    return transform

def image_process(network, content, style):
    C_size = content.size[::-1]
    S_size = style.size[::-1]
    
    content_tf1, resized = content_transform(C_size)
    content_frame = content_tf1(content)

    style_tf1 = style_transform(S_size)
    style = style_tf1(style)

    style = style.to(device).unsqueeze(0)
    content = content_frame.to(device).unsqueeze(0)
    
    with torch.no_grad():
        output = network(content, style)
    output = output.squeeze(0)
    
    if resized:
        resize_tf = resize_transform(C_size)
        output = resize_tf(output)
    
    return output.cpu()

def process_image(network, content_path, style_path, outfile):
    image_name = outfile + '/{:s}_stylized_{:s}.jpg'.format(
        splitext(basename(content_path))[0], splitext(basename(style_path))[0])
    content = Image.open(content_path).convert("RGB")
    style = Image.open(style_path).convert("RGB")
    output = image_process(network, content, style)
    save_image(output, image_name)
            
def test_image(network, content_paths, style_paths, output_path):
    pbar = tqdm(total = len(content_paths)*len(style_paths))
    for style_path in style_paths:
        for content_path in content_paths: 
            
            outfile = output_path + '/' + splitext(basename(content_path))[0] + '/'
            if not os.path.exists(outfile):
                os.makedirs(outfile)
                
            process_image(network, content_path, style_path, outfile)
            pbar.update(1)

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--KC', type=int, default=4)
    parser.add_argument('--KS', type=int, default=-10)
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save the output image(s)')
    parser.add_argument('--csbnet_path', type=str, default='./models/csbnet.pth',
                        help='The path of the csbnet pretrained model')
    parser.add_argument('--vgg_path', type=str, default='./models/vgg_normalised.pth',
                        help='The path of the pretained vgg-net model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = create_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #load pre-trained vgg-net
    vgg = net.vgg
    vgg.to(device)
    vgg.load_state_dict(torch.load(args.vgg_path, map_location=device))
    vgg.eval()
    
    
    #load CSBNet
    network = net.Net(vgg, KC=args.KC, KS=args.KS)
    network.to(device)
    network.csbnet.load_state_dict(torch.load(args.csbnet_path, map_location=device))
    network.eval()
    
    #inference
    content_paths, style_paths = load_images(args.content_dir, args.style_dir)
    test_image(network, content_paths, style_paths, args.output_dir)

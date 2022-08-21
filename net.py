import torch.nn as nn
import torch
from function import normal
from function import calc_mean_std
import scipy.stats as stats
from torchvision.utils import save_image

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

    
class Blending_Module(nn.Module):
    def __init__(self, in_dim):
        super(Blending_Module, self).__init__()
        self.J = nn.Conv2d(in_dim , in_dim, (1,1))
        self.K = nn.Conv2d(in_dim , in_dim, (1,1))
        self.W = nn.Conv2d(in_dim , in_dim, (1,1))
        self.R = nn.Conv2d(in_dim , in_dim, (1,1))
        
    def forward(self, content_enhance, style_enhance):
        Fc_tilde  = self.J(normal(content_enhance))
        B,C,H,W = style_enhance.size()
        Fs_tilde =  self.K(normal(style_enhance)).view(B,C,H*W)
        
        Gram_sum = Fs_tilde.sum(-1).view(B,C,1)
        
        Gram_s = (Fs_tilde @ Fs_tilde.permute(0,2,1) / Gram_sum).view(B,C,C,1)
        
        #Weight Gram Matrix
        Weighted_Gram = self.W(Gram_s).view(B,C,C)
        #get C weighted value
        sigma = torch.diagonal(Weighted_Gram,dim1=-2,dim2=-1).view(B,C,1,1) 
        Fcs = self.R(Fc_tilde * sigma + content_enhance)
        return Fcs

class CrSp_Module(nn.Module):
    def __init__(self, in_dim, K, type):
        super(CrSp_Module, self).__init__()
        self.f = nn.Conv2d(in_dim , in_dim , (1,1), groups=in_dim)
        self.g = nn.Conv2d(in_dim , in_dim , (1,1), groups=in_dim)
        self.K = K
        self.type = type
        
    def Crystallization(self, input):
        if self.type == 'style':
            B,C,H,W = input.shape
            input_zipped = input.view(-1,C,H*W)
            input_average = input_zipped.mean(dim=2).view(-1,C,1) #B*H*1
            input_center = input_zipped - input_average
            U, Sigma, V = torch.svd(input_center)
            VT=V.permute(0,2,1)
            temp = (U[:, :,0:self.K] @ torch.diag_embed(Sigma[:, 0:self.K]) @ VT[:, 0:self.K,:] + input_average).view(B,C,H,W)
            return self.g(temp)
    
        elif self.type == 'content':
            B,C,H,W = input.shape
            input_zipped = input.view(-1,C,H*W)
            input_average = input_zipped.mean(dim=2).view(-1,C,1)
            input_center = input_zipped - input_average          
            U,Sigma,V = torch.svd(input_center)
            VT=V.permute(0,2,1)
            temp = (U[:, :,self.K:] @ torch.diag_embed(Sigma[:, self.K:]) @ VT[:, self.K:,:] + input_average).view(B,C,H,W)
            return self.g(temp)
            
    def forward(self, content_feat):
        feature_globe = self.Crystallization(content_feat)
        feature_ori = self.f(content_feat)
        return feature_globe + feature_ori
    
class CSBNet(nn.Module):
    def __init__(self, in_dim, KC, KS):
        super(CSBNet, self).__init__()
        self.crsp_c = CrSp_Module(in_dim, KC, type='content')
        self.crsp_s = CrSp_Module(in_dim, KS, type='style')
        self.blending_module = Blending_Module(512)
        self.decoder = decoder
        
    def forward(self, content, style):
        Fc_enhanced = self.crsp_c(content)
        Fs_enhanced = self.crsp_s(style)
        Fcs = self.blending_module(Fc_enhanced, Fs_enhanced)
        return self.decoder(Fcs)
        
        
class Net(nn.Module):
    def __init__(self, encoder, KC, KS):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        self.csbnet = CSBNet(512, KC, KS)
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
                
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def forward(self, content, style):

        content_feats = self.encode_with_intermediate(content)
        style_feats = self.encode_with_intermediate(style)
        
        Ics = self.csbnet(content_feats[-2], style_feats[-2])
        return Ics

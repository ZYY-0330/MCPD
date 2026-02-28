import torch
import torch.nn as nn
from torchvision import models
from torch.cuda.amp import autocast
from configs.dataset_config import *
from torch.utils.data import DataLoader
import h5py

import numpy as np
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from configs.dataset_config import *
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from configs.dataset_config import *




class InvertedDualAttentionBlock(nn.Module):
 
    def __init__(self, in_channels, expansion_ratio=2):
        super().__init__()
        hidden_dim = in_channels * expansion_ratio
        
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, 
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
       
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1), 
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1), 
            nn.Sigmoid()
        )

     
        self.spatial_gate = nn.Sequential(
            
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        identity = x
       
        x = self.expand_conv(x)
        x = self.dw_conv(x)
        
        chan_att = self.channel_gate(x)
        x = x * chan_att
       
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_out, avg_out], dim=1)
        
        spatial_att = self.spatial_gate(spatial_input)
        x = x * spatial_att
        
       
        x = self.project_conv(x)
       
        return x + identity

class FeatureExtractionModel(nn.Module):
    def __init__(self, base_channels=512, device='cuda'):
        super().__init__()
        self.device = device
        
      
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.ModuleDict({
            'conv1': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
            'maxpool': resnet.maxpool,
            'layer1': resnet.layer1,
            'layer2': resnet.layer2,
            'layer3': resnet.layer3,
            'layer4': resnet.layer4
        })
        for param in self.backbone.parameters():
            param.requires_grad = False

      
        
        target_dim = 256
        self.target_dim = 256
        self.proc_layer1 = nn.Sequential(
            nn.InstanceNorm2d(256, affine=True), 
            nn.Conv2d(256, target_dim, 1), 
            nn.BatchNorm2d(target_dim), 
            nn.LeakyReLU(0.1, inplace=True)    
        )
        

        self.proc_layer2 = nn.Sequential(
            nn.InstanceNorm2d(512, affine=True), 
            nn.Conv2d(512, target_dim, 1), 
            nn.BatchNorm2d(target_dim), 
            nn.LeakyReLU(0.1, inplace=True)      
        )
        
      
        self.proc_layer3 = nn.Sequential(
            nn.InstanceNorm2d(1024, affine=True), 
            nn.Conv2d(1024, target_dim, 1), 
            nn.BatchNorm2d(target_dim), 
            nn.LeakyReLU(0.1, inplace=True)     
        )

        self.proc_layer4 = nn.Sequential(
            nn.InstanceNorm2d(2048, affine=True), 
            nn.Conv2d(2048, target_dim, 1), 
            nn.BatchNorm2d(target_dim), 
            nn.LeakyReLU(0.1, inplace=True)    
        )
        self.att1 = InvertedDualAttentionBlock(self.target_dim, expansion_ratio=2)
        self.att2 = InvertedDualAttentionBlock(self.target_dim, expansion_ratio=2)
        self.att3 = InvertedDualAttentionBlock(self.target_dim, expansion_ratio=2)
        self.att4 = InvertedDualAttentionBlock(self.target_dim, expansion_ratio=2)

        
        self.gate4to3 = nn.Conv2d(self.target_dim, 1, kernel_size=1) 
        self.gate3to2 = nn.Conv2d(self.target_dim, 1, kernel_size=1)
        self.gate2to1 = nn.Conv2d(self.target_dim, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): 
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        if self.training:
            x = [feat + torch.randn_like(feat) * 0.01 for feat in x]

        f1 = self.proc_layer1(x[0])
        f2 = self.proc_layer2(x[1])
        f3 = self.proc_layer3(x[2])
        f4 = self.proc_layer4(x[3])

        f1 = self.att1(f1)
        f2 = self.att2(f2)
        f3 = self.att3(f3)
        f4 = self.att4(f4)

       
        mask4 = torch.sigmoid(self.gate4to3(f4)) 
        f3 = f3 * mask4 + f3 
        
       
        mask3 = torch.sigmoid(self.gate3to2(f3))
        f2 = f2 * mask3 + f2
        
        mask2 = torch.sigmoid(self.gate2to1(f2))
        f1 = f1 * mask2 + f1

        seq1 = f1.flatten(2).transpose(1, 2)
        seq2 = f2.flatten(2).transpose(1, 2)
        seq3 = f3.flatten(2).transpose(1, 2)
        seq4 = f4.flatten(2).transpose(1, 2)

        return [seq2, seq3,seq4]
   
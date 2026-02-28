import torch
import torch.nn as nn
from  BERT import MathBERTTextFeatureExtractor
from  RestNet import FeatureExtractionModel
from  fusion_model import HierarchicalFusionSystem
import time
import gc
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

class EndToEndContrastiveModel(nn.Module):
    def __init__(self, device='auto'):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == 'auto' else device
        )
        self.img_feature = FeatureExtractionModel().float().to(self.device) 
        self.fusion = HierarchicalFusionSystem().float().to(self.device)
        nn.init.xavier_uniform_(self.img_proj.weight)
        if self.img_proj.bias is not None:
            nn.init.constant_(self.img_proj.bias, 0)

        self.to(self.device)
    def forward(self, img_raw_list, txt_raw_list,padding_mask):

        img_feats= self.img_feature(img_raw_list)
        fused_feat, img_rep, txt_rep = self.fusion(txt_raw_list, img_feats, padding_mask)  # 直接传入mask
        return  fused_feat, img_rep, txt_rep

    
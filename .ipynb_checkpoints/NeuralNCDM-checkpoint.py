
import torch.nn as nn
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
import time
import torch.nn.functional as F
logger = logging.getLogger(__name__)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset import RecordDataset, RelationBuilder,FullDataset
from configs.dataset_config import *
import json
import pandas as pd
from EndToEndContrastiveModel import EndToEndContrastiveModel
class Net(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, problem_dataset):

        super(Net, self).__init__()
        
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(input_dim=knowledge_n+1024,out_dim=knowledge_n) 
        

        
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.exer_feat = nn.Embedding(self.exer_n, 2)
        self.student_emb_text = nn.Embedding(self.emb_num, self.stu_dim)
        self.student_emb_img = nn.Embedding(self.emb_num, self.stu_dim)
      


     
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        
        self.pre = nn.Linear(self.knowledge_dim,1)
        self.problem_dataset = problem_dataset
     
       
        self.to(self.device)

        self.model_feat = EndToEndContrastiveModel().to(self.device)
        
       
        
        self.problem_mapper = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
           
        )
        self.knowledge_mapper = torch.nn.Linear(1024, 512)

        self.kc_importance = nn.Parameter(torch.full((1, self.knowledge_dim), 0.5))

        self.kc_importance_layer = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

       

        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

       
        self.problem_768 = nn.Linear(768,512)
        self.freq_amplifier = nn.Parameter(torch.tensor(10.0)) 
        self.base_scale = nn.Parameter(torch.tensor(6.0))       
        self.freq_power = nn.Parameter(torch.tensor(0.5))    
        
        self.diff_item_proj = nn.Linear(1024, 512) 
        self.diff_knowledge_emb = nn.Embedding(knowledge_n, 512)  
        self.diff_scale = nn.Parameter(torch.tensor(1.0))  
        self.diff_bias = nn.Parameter(torch.tensor(0.0))   
        
        self.disc_item_proj = nn.Linear(1024, 512) 
        self.disc_knowledge_emb = nn.Embedding(knowledge_n, 512)  
        self.disc_scale = nn.Parameter(torch.tensor(1.0)) 
        self.disc_bias = nn.Parameter(torch.tensor(0.0))
        
        
        self.problem = nn.Embedding(self.exer_n, 1024)

        self.problem_text = nn.Embedding(self.exer_n, 512)
        self.problem_img = nn.Embedding(self.exer_n, 512)
        self.feature_adapter = nn.Sequential(
            nn.Conv1d(512, self.C, kernel_size=1), 
            nn.AdaptiveMaxPool1d(self.M)
        )
        
        self.output_layer = nn.Linear(knowledge_n, 1)
        self.feature_weights = nn.Parameter(torch.randn(329)) 
       
        self.FEATURE_DIM = 256 
        self.GATE_HIDDEN_DIM = 128   
        self.DIFF_HIDDEN_DIM = 128   
        self.LATENT_K_DIM = 256     

      
        self.student_emb = nn.Embedding(student_n, self.knowledge_dim)
        self.k_difficulty_NCDM = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination_NCDM = nn.Embedding(self.exer_n, 1)

        self.W_p = nn.Parameter(torch.randn(self.FEATURE_DIM, self.LATENT_K_DIM) * 0.02)
        
        self.fusion_dropout = nn.Dropout(p=0.5) 
        self.diff_head_k = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, self.DIFF_HIDDEN_DIM), # ✅ 768 -> 384
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.DIFF_HIDDEN_DIM, 1)
        )
        self.norm = nn.LayerNorm(self.FEATURE_DIM * 2)
       
        self.gate_net = nn.Sequential(
            nn.Linear(self.FEATURE_DIM * 2, self.GATE_HIDDEN_DIM), # ✅ 1536 -> 128
            nn.ReLU(),
            nn.Linear(self.GATE_HIDDEN_DIM, 1)
        )
     
        self.output_layer = nn.Linear(knowledge_n, 1)


        cache = torch.load(OUTPUT_FILE, map_location='cpu') 
      
        sorted_pids = sorted(list(cache.keys()))

       
        self.register_buffer('bank_img_l1', torch.stack([cache[p]['img'][0] for p in sorted_pids]))
        self.register_buffer('bank_img_l2', torch.stack([cache[p]['img'][1] for p in sorted_pids]))
        self.register_buffer('bank_img_l3', torch.stack([cache[p]['img'][2] for p in sorted_pids]))
        self.register_buffer('bank_img_l4', torch.stack([cache[p]['img'][3] for p in sorted_pids]))
        
     
        self.register_buffer('bank_txt_l1', torch.stack([cache[p]['txt'][0] for p in sorted_pids]))
        self.register_buffer('bank_txt_l2', torch.stack([cache[p]['txt'][1] for p in sorted_pids]))
        self.register_buffer('bank_txt_l3', torch.stack([cache[p]['txt'][2] for p in sorted_pids]))
        
       
        if 'mask' in cache[sorted_pids[0]]:

            self.register_buffer('bank_mask', torch.stack([cache[p]['mask'] for p in sorted_pids]))
        else:
            self.bank_mask = None
            
       

        self.snr_diff_head = SNRDifficultyHead(feature_dim=256) # 确保维度是 256



        self.know_bert_emb = nn.Embedding.from_pretrained(pretrained_matrix, freeze=True)
        
        self.know_projector = nn.Linear(768, 256)

       
        self.aux_classifier = nn.Linear(256, self.know_bert_emb.num_embeddings) # 86
        
        self.se_gate = nn.Sequential(
            nn.Linear(self.knowledge_dim, self.knowledge_dim // 2),
            nn.LayerNorm(self.knowledge_dim // 2),
            nn.ReLU(),
            nn.Linear(self.knowledge_dim // 2, self.knowledge_dim)
           
        )
        
        self.alpha_net = nn.Linear(self.knowledge_dim, 1)
        self.diff_head_global = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, self.DIFF_HIDDEN_DIM), # ✅ 768 -> 384
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.DIFF_HIDDEN_DIM, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
     
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m is self.prednet_full3 or m in [self.diff_head[-1], self.disc_head[-1]]:
                        nn.init.constant_(m.bias, -1.0)
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                if hasattr(m, "is_knowledge_emb") and m.is_knowledge_emb:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                else:
                    nn.init.uniform_(m.weight, -0.05, 0.05)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
     
        nn.init.xavier_normal_(self.W_p)
        nn.init.normal_(self.W_p, mean=0, std=0.05)

    def get_knowledge_embedding(self, knowledge_ids):

        bert_feats = self.know_bert_emb(knowledge_ids) 

        final_feats = self.know_projector(bert_feats)
        
        return final_feats
    def forward(self, batch):
       
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            device = torch.device('cpu')
        
        
        stu_id = batch['student_ids'].long().to(device)
        exer_id = batch['problem_ids'].long().to(device)
        kn_emb = batch['knowledges'].to(device)


        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        e_discrimination = torch.sigmoid(self.e_discrimination_NCDM(exer_id)) * 10
        k_difficulty = torch.sigmoid(self.k_difficulty_NCDM(exer_id))
        
        pids = batch['problem_ids'].long().to(self.bank_img_l1.device)
        
     
        img_raw_list = [
            self.bank_img_l1[pids],
            self.bank_img_l2[pids],
            self.bank_img_l3[pids],
            self.bank_img_l4[pids]
        ]
        
        txt_raw_list = [
            self.bank_txt_l1[pids],
            self.bank_txt_l2[pids],
            self.bank_txt_l3[pids]
        ]
        
      
        if self.bank_mask is not None:
            raw_mask = self.bank_mask[pids]
            padding_mask = (raw_mask == 0)
        else:
            padding_mask = None
       
        unique_pids, inverse_indices = torch.unique(exer_id, sorted=True, return_inverse=True)
        
       
        perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
       
        unique_indices = perm.new_empty(unique_pids.size(0)).scatter_(0, inverse_indices, perm)
       
        unique_img_raw = [t[unique_indices] for t in img_raw_list] 
        unique_txt_raw = [t[unique_indices] for t in txt_raw_list]
        
        if padding_mask is not None:
            unique_padding_mask = padding_mask[unique_indices]
        else:
            unique_padding_mask = None

        
        unique_kn_labels = kn_emb[unique_indices].float() 

        unique_fused_feat , final_img_rep, final_txt_rep= self.update_features(unique_img_raw, unique_txt_raw, unique_padding_mask)

        problem_feat = unique_fused_feat[inverse_indices]

        mse_loss = torch.tensor(0.0, device=device)
        problem_feat = F.layer_norm(problem_feat, problem_feat.shape[-1:])
        F_j = problem_feat # [B, N, D]
       
        batch_indices, knowledge_indices = torch.nonzero(kn_emb, as_tuple=True)
        batch_size = kn_emb.shape[0]
        total_knowledge = self.know_bert_emb.num_embeddings
        modality_k_difficulty = torch.zeros(batch_size, total_knowledge, device=device)
        if len(batch_indices) > 0:
            with torch.cuda.amp.autocast(enabled=False):
                W_p_safe = self.W_p.float()
                raw_bert_emb = self.know_bert_emb(knowledge_indices) 
                selected_knowledge = self.know_projector(raw_bert_emb) 
                selected_knowledge = F.normalize(selected_knowledge, p=2, dim=-1)
                intermediate = torch.matmul(selected_knowledge, W_p_safe)
                intermediate = F.layer_norm(intermediate, intermediate.shape[-1:])
                
                selected_F_j = F_j[batch_indices].float()
                
                W_j_selected = torch.bmm(selected_F_j, intermediate.unsqueeze(-1)).squeeze(-1)
                U_j_selected = torch.bmm(W_j_selected.unsqueeze(1), selected_F_j).squeeze(1)
                U_j_selected = F.layer_norm(U_j_selected, U_j_selected.shape[-1:])
                
                linear_output = self.diff_head_k(U_j_selected)
                selected_difficulty_pred = torch.sigmoid(linear_output)

            selected_difficulty_pred = selected_difficulty_pred.to(modality_k_difficulty.dtype)
            modality_k_difficulty[batch_indices, knowledge_indices] = selected_difficulty_pred.squeeze(1)


        f_k_difficulty =1.0* modality_k_difficulty + 0.0* k_difficulty
        
        stu_raw = self.student_emb(stu_id)     
        raw_interaction = stu_raw * f_k_difficulty
       
        gate_logits = self.se_gate(f_k_difficulty)
      
        channel_weights = torch.sigmoid(gate_logits * 5.0)
        
        clean_interaction = raw_interaction * channel_weights
        
        alpha_logit = self.alpha_net(clean_interaction)
        alpha_sensitivity = 1.0 + 0.4 * torch.tanh(alpha_logit/2.0)
        

        loss_reg = torch.mean(torch.abs(channel_weights - 0.5)) * -0.01 + torch.mean(alpha_logit ** 2) * 0.01
       
        core_term = stu_emb - f_k_difficulty
        input_x_final = e_discrimination * (core_term * alpha_sensitivity) * kn_emb
     
        input_x_final = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_final)))
        input_x_final = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_final)))
        pred_final = self.prednet_full3(input_x_final)

        pred_final = torch.clamp(pred_final, min=-10.0, max=10.0) # 防爆

        return pred_final, loss_reg, loss_reg, loss_reg
        
 
        
   
    def update_features(self, img_raw_list, txt_raw_list,padding_mask):
        fused_out, final_img_rep, final_txt_rep = self.model_feat(img_raw_list, txt_raw_list,padding_mask)
        return fused_out, final_img_rep, final_txt_rep

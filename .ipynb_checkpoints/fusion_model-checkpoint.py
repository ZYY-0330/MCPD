import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAdaptiveFusion(nn.Module):
    def __init__(self, dim=256, num_scales=3):
        super().__init__()

        self.scale_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_scales)])
        self.score_net = nn.Sequential(
            nn.Linear(dim * 2, dim // 4), 
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(dim // 4, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, feats_list):
        
        normed_feats = [self.scale_norms[i](f) for i, f in enumerate(feats_list)]
        stack = torch.stack(normed_feats, dim=1) # [B, 3, N, C]
        
        B, n_scales, N, C = stack.shape
        
        avg_p = torch.mean(stack, dim=2)
        max_p = torch.max(stack, dim=2)[0]
        
        scores = self.score_net(torch.cat([avg_p, max_p], dim=2).view(-1, C*2))
        scores = scores.view(B, n_scales)
        
      
        weights_raw = self.softmax(scores / 5.0)
        
        
        weights_expanded = weights_raw.view(B, n_scales, 1, 1)

        fused_feat = torch.sum(stack * weights_expanded, dim=1)
        
        return self.final_norm(fused_feat), weights_raw

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist

def init_weights_safe(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.005) 
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.q_proj.apply(init_weights_safe)
        self.k_proj.apply(init_weights_safe)
        self.v_proj.apply(init_weights_safe)
        self.out_proj.apply(init_weights_safe)
      
        
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, temp_scale=1.0, return_weights=False):
        B, Lq, _ = query.shape
        B, Lk, _ = key.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q * temp_scale

        attn_weights = None

        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = (mask == 0).view(B, 1, 1, Lk).expand(B, self.num_heads, Lq, Lk)

        attn_weights = None

      
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
       
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        out = self.out_proj(attn_output)
        out = self.proj_dropout(out)
        
        return out, attn_weights
    def _stable_attention(self, Q, K, V, mask):
        
        d_k = self.head_dim
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0].detach() 
        scores_stable = scores - scores_max 

        attn_weights = F.softmax(scores_stable, dim=-1)
        
        p_attn = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(p_attn, V)
        
        return attn_output, attn_weights

class SharedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = CustomMultiHeadAttention(dim, num_heads, dropout)

    def forward(self, x, mask=None, temp=1.0, return_weights=False):
        out, weights = self.attn(
            query=x, key=x, value=x, 
            mask=mask, temp_scale=temp, 
            return_weights=return_weights
        )
        return x + out, weights

class SharedBiModalFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn_i2t = CustomMultiHeadAttention(dim, num_heads, dropout)
        self.attn_t2i = CustomMultiHeadAttention(dim, num_heads, dropout)
        
        self.injection_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, img, text, text_mask=None, temp=1.0, return_weights=False):
        i2t_out, weights_i2t = self.attn_i2t(
            img, text, text, mask=text_mask, temp_scale=temp, return_weights=return_weights
        )
        img_enriched = img + i2t_out
        
        t2i_out, weights_t2i = self.attn_t2i(
            text, img, img, mask=None, temp_scale=temp, return_weights=return_weights
        )
        text_enriched = text + t2i_out
        
        text_context = text_enriched.max(dim=1, keepdim=True)[0]
        text_context_expanded = text_context.expand(-1, img.shape[1], -1)
        
        concat = torch.cat([img_enriched, text_context_expanded], dim=-1)
        gate = self.injection_gate(concat)
        
        output = img_enriched * (1 - gate) + text_context_expanded * gate
        
        return output, (weights_i2t, weights_t2i)

class HierarchicalFusionSystem(nn.Module):
    def __init__(self, text_dim=768, img_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(text_dim, img_dim), nn.LayerNorm(img_dim), nn.ReLU())
            for _ in range(3)
        ])

    
        self.shared_img_attn = SharedSelfAttention(img_dim, num_heads, dropout)
        self.shared_text_attn = SharedSelfAttention(img_dim, num_heads, dropout)
        self.shared_cross_attn = SharedBiModalFusion(img_dim, num_heads, dropout)

       
        self.norms_img_self = nn.ModuleList([nn.LayerNorm(img_dim) for _ in range(3)])
        self.norms_text_self = nn.ModuleList([nn.LayerNorm(img_dim) for _ in range(3)])
        self.norms_img_cross = nn.ModuleList([nn.LayerNorm(img_dim) for _ in range(3)])
        self.norms_text_cross = nn.ModuleList([nn.LayerNorm(img_dim) for _ in range(3)])

        
        init_values = torch.tensor([4.0,3.5,3.0]) 
        self.temp_img = nn.Parameter(init_values.clone()) 
        self.temp_txt = nn.Parameter(init_values.clone()) 
        self.temp_cross = nn.Parameter(init_values.clone()) 
        self.fused = ScaleAdaptiveFusion(img_dim, num_scales=3)

        self.sum  = 0

    def forward(self, text_feats, img_feats, text_mask=None):

        self.sum = self.sum+1
        return_debug = False
        fused_outputs = []
        sum_c = 1000
        
        for i in range(3):
        
            curr_text = self.projs[i](text_feats[i])
            curr_img = img_feats[i]
            
            t_img = F.softplus(self.temp_img[i])
            t_txt = F.softplus(self.temp_txt[i])
            t_cross = F.softplus(self.temp_cross[i])
            
            img_in = self.norms_img_self[i](curr_img)
            text_in = self.norms_text_self[i](curr_text)

            curr_img, w_img = self.shared_img_attn(
                img_in, mask=None, temp=t_img, return_weights=return_debug
            )
            
            curr_text, w_txt = self.shared_text_attn(
                text_in, mask=text_mask, temp=t_txt, return_weights=return_debug
            )
            
            img_cross_in = self.norms_img_cross[i](curr_img)
            text_cross_in = self.norms_text_cross[i](curr_text)
           
            fused_layer, (w_i2t, w_t2i) = self.shared_cross_attn(
                img_cross_in, text_cross_in, text_mask, temp=t_cross,
                return_weights=return_debug
            )
          
            fused_outputs.append(fused_layer)

        final_out, scale_weights = self.fused(fused_outputs)
        
        return final_out, None, None

    
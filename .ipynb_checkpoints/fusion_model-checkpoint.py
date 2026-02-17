import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAdaptiveFusion(nn.Module):
    def __init__(self, dim=256, num_scales=3):
        super().__init__()
        
        # 🔥 修改 1: 尺度对齐 Norm
        # 防止 Layer 0 因为数值大而天然占优，强迫它们在同一起跑线竞争
        self.scale_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_scales)])
        
        self.score_net = nn.Sequential(
            nn.Linear(dim * 2, dim // 4), 
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(dim // 4, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        
        # 🔥 修改 2: 最终输出稳压 Norm (对齐 A30/4090 差异的关键)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, feats_list):
        # 1. 先对齐各个尺度的特征分布
        normed_feats = [self.scale_norms[i](f) for i, f in enumerate(feats_list)]
        stack = torch.stack(normed_feats, dim=1) # [B, 3, N, C]
        
        B, n_scales, N, C = stack.shape
        
        # 2. 计算统计量
        avg_p = torch.mean(stack, dim=2)
        max_p = torch.max(stack, dim=2)[0]
        
        # 3. 打分
        scores = self.score_net(torch.cat([avg_p, max_p], dim=2).view(-1, C*2))
        scores = scores.view(B, n_scales)
        
        # 🔥 修改 3: 高温 Softmax (T=5.0)
        # 强行拉平权重分布，防止出现 [0.99, 0, 0] 这种极端分布
        weights_raw = self.softmax(scores / 5.0)
        
        # 🔥 修改 4: 尺度 Dropout (仅训练时)
        # 随机把某个尺度的权重扔掉，逼模型学会用 Layer 1 和 2
        if self.training:
            # 10% 的概率丢弃某个尺度
            scale_mask = (torch.rand(B, n_scales, 1, 1, device=stack.device) > 0.1).float()
            weights_expanded = weights_raw.view(B, n_scales, 1, 1) * scale_mask
            # 重新归一化防止全0
            weights_expanded = weights_expanded / (weights_expanded.sum(dim=1, keepdim=True) + 1e-6)
        else:
            weights_expanded = weights_raw.view(B, n_scales, 1, 1)

        # 4. 融合
        fused_feat = torch.sum(stack * weights_expanded, dim=1)
        
        # 5. 返回 Norm 后的特征
        return self.final_norm(fused_feat), weights_raw

import torch
import torch.nn as nn
import torch.nn.functional as F
'''
# ====================================================================
# 1. 基础 Attention 模块 (支持 FlashAttention + 外部温度控制)
# ====================================================================
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
        
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, temp_scale=1.0):
        """
        temp_scale: 温度缩放系数。
                    > 1.0 : 让注意力更尖锐 (适合深层)
                    < 1.0 : 让注意力更平滑
        """
        B, Lq, _ = query.shape
        B, Lk, _ = key.shape

        # 1. 投影 + 分头 [B, Heads, Len, HeadDim]
        Q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 🔥 关键：应用温度缩放
        # F.sdpa 默认缩放是 1/sqrt(d)。我们乘上 temp_scale，等效于公式中的 Q*K / (sqrt(d) * T)
        # 注意：这里我们只缩放 Q 即可
        Q = Q * temp_scale 

        # 3. 处理 Mask (适配 FlashAttention 的 4D 格式)
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # [B, Lk] -> [B, 1, 1, Lk] (True=Padding)
                # FlashAttention 的 mask 要求: True 表示要 mask 掉的位置 (padding)
                # 如果你的 mask 是 1=有效 0=padding，那么这里要用 (mask==0)
                attn_mask = (mask == 0).view(B, 1, 1, Lk).expand(B, self.num_heads, Lq, Lk)
            # 如果 mask 已经是 4D bool，直接用

        # 4. 调用 FlashAttention (极速、省显存)
        # is_causal=False (因为这不是 GPT 生成任务)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        # 5. 重组输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        out = self.out_proj(attn_output)
        out = self.proj_dropout(out)
        
        # 注意：FlashAttention 不返回 weights，返回 None
        return out, None
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist

def init_weights_safe(m):
    """强制使用极小的标准差初始化投影层"""
    if isinstance(m, nn.Linear):
        # 0.005 是一个经验值，非常保守，适合 Attention 机制
        nn.init.normal_(m.weight, mean=0.0, std=0.005) 
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
# ====================================================================
# 1. 基础 Attention 模块 (支持 FlashAttention + 调试模式)
# ====================================================================
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

        # 🔥 关键修复：应用安全初始化
        self.q_proj.apply(init_weights_safe)
        self.k_proj.apply(init_weights_safe)
        self.v_proj.apply(init_weights_safe)
        self.out_proj.apply(init_weights_safe)
      
        
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, temp_scale=1.0, return_weights=False):
        B, Lq, _ = query.shape
        B, Lk, _ = key.shape
        
       
        # 1. 投影 + 分头
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
      
            
        # 分头
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        
      
        
        # 2. 应用温度缩放
        Q = Q * temp_scale
        
      
        attn_weights = None

        # 3. 处理 Mask (FlashAttention 格式)
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # [B, Lk] -> [B, 1, 1, Lk]
                attn_mask = (mask == 0).view(B, 1, 1, Lk).expand(B, self.num_heads, Lq, Lk)

        attn_weights = None

        # ============================================================
        # 🔄 分支路口：极速模式 vs 调试模式
        # ============================================================
        if not return_weights:
            # --- 方案 A: 极速模式 (FlashAttention) ---
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            # 🔥 关键修复：直接使用torch内置的稳定注意力
            # 这个方法比手动计算稳定得多
            attn_output, attn_weights = self._stable_attention(
                Q, K, V, attn_mask
            )

        # 5. 重组输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        out = self.out_proj(attn_output)
        out = self.proj_dropout(out)
        
        return out, attn_weights
    def _stable_attention(self, Q, K, V, mask):
        
        d_k = self.head_dim
        
        # 1. 计算分数 (Softmax 数值稳定技巧已经内置在你的手动代码里了)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        # 🔥 核心修复：直接使用你的手动稳定 Softmax 逻辑
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0].detach() 
        scores_stable = scores - scores_max 

        attn_weights = F.softmax(scores_stable, dim=-1)
        
        # Dropout
        p_attn = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 加权求和
        attn_output = torch.matmul(p_attn, V)
        
        return attn_output, attn_weights

# ====================================================================
# 2. 共享组件包装器 (支持透传 return_weights)
# ====================================================================
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
        # 1. 图像看文本 (I2T)
        i2t_out, weights_i2t = self.attn_i2t(
            img, text, text, mask=text_mask, temp_scale=temp, return_weights=return_weights
        )
        img_enriched = img + i2t_out
        
        # 2. 文本看图像 (T2I)
        t2i_out, weights_t2i = self.attn_t2i(
            text, img, img, mask=None, temp_scale=temp, return_weights=return_weights
        )
        text_enriched = text + t2i_out
        
        # 3. 注入
        text_context = text_enriched.max(dim=1, keepdim=True)[0]
        text_context_expanded = text_context.expand(-1, img.shape[1], -1)
        
        concat = torch.cat([img_enriched, text_context_expanded], dim=-1)
        gate = self.injection_gate(concat)
        
        output = img_enriched * (1 - gate) + text_context_expanded * gate
        
        return output, (weights_i2t, weights_t2i)

# ====================================================================
# 3. 🔥 层次化融合系统 (主类 - 增加独立温度和调试接口)
# ====================================================================
class AttentionWeightAnalyzer:
    """
    只基于权重分析注意力的有效性
    """
    
    @staticmethod
    def analyze_weights_simple(weights, temperature, layer_name=""):
        """详细分析注意力权重"""
        if weights is None:
            print(f"{layer_name}: 权重为None")
            return False, "无权重数据"
        
        print(f"\n🔍 {layer_name} 详细分析:")
        print(f"   权重形状: {weights.shape}")
        print(f"   温度: {temperature}")
        
        # 取第一个样本，第一个头的权重
        if weights.dim() == 4:  # [B, H, Lq, Lk]
            w = weights[0, 0].detach()
            print(f"   多头注意力，取头0")
        elif weights.dim() == 3:  # [B, Lq, Lk]
            w = weights[0].detach()
        else:
            print(f"   异常维度: {weights.dim()}")
            return False, f"权重维度异常"
        
        Lq, Lk = w.shape
        print(f"   查询长度Lq: {Lq}, 键长度Lk: {Lk}")
        
        # 检查是否有NaN/Inf
        if torch.isnan(w).any():
            print(f"   ⚠️ 警告: 权重包含NaN!")
        if torch.isinf(w).any():
            print(f"   ⚠️ 警告: 权重包含Inf!")
        
        # 检查权重是否全部相同
        w_flat = w.flatten()
        if (w_flat == w_flat[0]).all():
            print(f"   ⚠️ 警告: 所有权重都相同!")
            print(f"   权重值: {w_flat[0]:.6f}")
            return False, "所有权重相同"
        
        # 计算统计信息
        print(f"   权重最小值: {w.min():.6f}")
        print(f"   权重最大值: {w.max():.6f}")
        print(f"   权重均值: {w.mean():.6f}")
        print(f"   权重标准差: {w.std():.6f}")
        
        # 检查行和是否为1（softmax特性）
        row_sums = w.sum(dim=-1)
        row_sum_error = (row_sums - 1.0).abs().max()
        print(f"   行和最大误差: {row_sum_error:.6f}")
        
        # 计算集中度
        eps = 1e-10
        entropy = -(w * torch.log(w + eps)).sum(dim=-1).mean()
        max_entropy = math.log(Lk)
        concentration = 1 - (entropy / max_entropy).item()
        
        print(f"   熵: {entropy.item():.6f}")
        print(f"   最大熵（均匀分布）: {max_entropy:.6f}")
        print(f"   集中度: {concentration:.6f}")
        
        # 检查是否接近均匀分布
        uniform_value = 1.0 / Lk
        uniform_diff = (w - uniform_value).abs().mean()
        print(f"   与均匀分布的差异: {uniform_diff:.6f}")
        
        if concentration < 0.1:
            return False, f"注意力过于分散(集中度:{concentration:.3f})"
        elif concentration > 0.9:
            return False, f"注意力过于集中(集中度:{concentration:.3f})"
        else:
            return True, f"注意力正常(集中度:{concentration:.3f})"
class HierarchicalFusionSystem(nn.Module):
    def __init__(self, text_dim=768, img_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(text_dim, img_dim), nn.LayerNorm(img_dim), nn.ReLU())
            for _ in range(3)
        ])

        # --- 共享组件 ---
        self.shared_img_attn = SharedSelfAttention(img_dim, num_heads, dropout)
        self.shared_text_attn = SharedSelfAttention(img_dim, num_heads, dropout)
        self.shared_cross_attn = SharedBiModalFusion(img_dim, num_heads, dropout)

        # --- 独立 Norm ---
        self.norms_img_self = nn.ModuleList([nn.LayerNorm(img_dim) for _ in range(3)])
        self.norms_text_self = nn.ModuleList([nn.LayerNorm(img_dim) for _ in range(3)])
        self.norms_img_cross = nn.ModuleList([nn.LayerNorm(img_dim) for _ in range(3)])
        self.norms_text_cross = nn.ModuleList([nn.LayerNorm(img_dim) for _ in range(3)])

        # --- 🔥 独立温度参数 (9个) ---
        # 3个模态类型 x 3个层级
        # 初始化值建议：1.0 或 2.0 (根据你之前的实验，可以设高一点)
        init_values = torch.tensor([4.0,3.5,3.0]) 



        self.temp_img = nn.Parameter(init_values.clone()) 

        self.temp_txt = nn.Parameter(init_values.clone()) 

        self.temp_cross = nn.Parameter(init_values.clone()) 


        self.fused = ScaleAdaptiveFusion(img_dim, num_scales=3)

        self.sum  = 0
    '''
    def forward(self, text_feats, img_feats, text_mask=None):
        """
        HierarchicalFusionSystem (Single Layer Version)
        """
        fused_outputs = []
        
        # ♻️ 只运行一次 (Layer 0) - 或者你可以指定具体的某一层索引
        # 如果你想跑单层，通常取最后一层或者第一层，这里示例取第 0 层
        i = 2
        
        # 1. 投影层输出
        curr_text = self.projs[i](text_feats[i])
        curr_img = img_feats[i]
        
        # 获取当前层的温度 (softplus 保证 > 0)
        t_img = F.softplus(self.temp_img[i])
        t_txt = F.softplus(self.temp_txt[i])
        t_cross = F.softplus(self.temp_cross[i])
        
        # 2. LayerNorm 输出 (Attention 输入)
        img_in = self.norms_img_self[i](curr_img)
        text_in = self.norms_text_self[i](curr_text)

        # --- 步骤 1 & 2: 自注意力 ---
        curr_img, _ = self.shared_img_attn(
            img_in, mask=None, temp=t_img, return_weights=False
        )
        
        curr_text, _ = self.shared_text_attn(
            text_in, mask=text_mask, temp=t_txt, return_weights=False
        )
        
        # 计算 Final Representations (用于 Loss)
        # Global Average Pooling
        final_img_rep = curr_img.mean(dim=1) 
        
        # Masked Mean for Text
        if text_mask is not None:
             mask_broadcast = text_mask.unsqueeze(-1).float()
             final_txt_rep = (curr_text * mask_broadcast).sum(dim=1) / (mask_broadcast.sum(dim=1) + 1e-6)
        else:
             final_txt_rep = curr_text.mean(dim=1)

        # --- 步骤 3: 跨模态融合 ---
        img_cross_in = self.norms_img_cross[i](curr_img)
        text_cross_in = self.norms_text_cross[i](curr_text)
        
        fused_layer, _ = self.shared_cross_attn(
            img_cross_in, text_cross_in, text_mask, temp=t_cross,
            return_weights=False
        )
        
        #fused_outputs.append(fused_layer)

        # 最终融合 (虽然只有一层，但为了保持接口一致，还是过一下 fused 模块)
        # 如果 self.fused 是处理列表的，传入单元素列表即可
        #final_out, _ = self.fused(fused_outputs)

        return fused_layer, final_img_rep, final_txt_rep
    '''
    def forward(self, text_feats, img_feats, text_mask=None):
        """
        HierarchicalFusionSystem 的 forward 方法
        """
        self.sum = self.sum+1
        return_debug = False
        fused_outputs = []
        sum_c = 1000
        
        # ♻️ 循环 3 次 (Layer 0, 1, 2)
        for i in range(3):
        #for i in range(3): 
        #for i in range(2, 3):
        #for i in [2]:
            # 1. 投影层输出
            curr_text = self.projs[i](text_feats[i])
            curr_img = img_feats[i]
            
            # 获取当前层的温度 (softplus 保证 > 0)
            t_img = F.softplus(self.temp_img[i])
            t_txt = F.softplus(self.temp_txt[i])
            t_cross = F.softplus(self.temp_cross[i])
            
           

            # 2. LayerNorm 输出 (Attention 输入)
            img_in = self.norms_img_self[i](curr_img)
            text_in = self.norms_text_self[i](curr_text)

           
            
           
            curr_img, w_img = self.shared_img_attn(
                img_in, mask=None, temp=t_img, return_weights=return_debug
            )
            
          
            
            curr_text, w_txt = self.shared_text_attn(
                text_in, mask=text_mask, temp=t_txt, return_weights=return_debug
            )
            
            

            
            # --- 步骤 3: 跨模态融合 ---
            img_cross_in = self.norms_img_cross[i](curr_img)
            text_cross_in = self.norms_text_cross[i](curr_text)
           
            fused_layer, (w_i2t, w_t2i) = self.shared_cross_attn(
                img_cross_in, text_cross_in, text_mask, temp=t_cross,
                return_weights=return_debug
            )
          
            
                
           
            fused_outputs.append(fused_layer)
            

        # 最终融合
        final_out, scale_weights = self.fused(fused_outputs)
        
        return final_out, None, None

    
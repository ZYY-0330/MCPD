
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


class Generator(nn.Module):
    def __init__(self, input_dim=658, out_dim=329, max_adjust=0.2):
        super().__init__()
        '''
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            #nn.LayerNorm(512),          # 👈 加在这里
            #nn.Linear(512, 512),
            #nn.ReLU(0.2),
            #nn.Dropout(0.5),  # 新增：输入层后立即Dropout
            nn.Linear(512,out_dim),
            nn.LeakyReLU(0.2),
            #nn.Linear(128, out_dim),
            #nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Tanh()  # 输出范围是 [-1, 1]
        )
        
        #self.max_adjust = nn.Parameter(torch.tensor(0.2))  # 会自动优化
        self.max_adjust = 0.5 # 会自动优化
        '''
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # 减少参数量
            nn.LayerNorm(256),          # 恢复 LayerNorm
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),            # 增强 Dropout
            nn.Linear(256, out_dim),
            #nn.Tanh()
        )
        self.max_adjust = 1.0
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # Xavier 保持输入输出方差稳定
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, stu_emb,difficulty_input):
        x = torch.cat([stu_emb,difficulty_input], dim=1)
        adjustment = self.net(x) * self.max_adjust  # 允许调节幅度
       
        return adjustment

import time
import torch  
from concurrent.futures import ThreadPoolExecutor
import gc
import os
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)



import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDifficulty(nn.Module):
    """
    输入: X (B, N, L)
    输出: difficulty (B, M) 0~1
    每个知识概念独立关注题目的N个角度
    仅针对涉及的知识概念学习权重
    """
    def __init__(self, L, M):
        super(KnowledgeDifficulty, self).__init__()
        self.L = L      # 每个角度的特征维度
        self.M = M      # 知识概念数量

        # 学习每个知识概念对每个角度的权重
        self.angle_attn = nn.Linear(L, M)  # L -> M

        # 将加权后的特征映射成标量难度
        self.to_scalar = nn.Linear(L, 1)
                # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # angle_attn: Xavier 初始化更合适
        nn.init.xavier_uniform_(self.angle_attn.weight)
        nn.init.constant_(self.angle_attn.bias, 0.)

        # to_scalar: Kaiming 也可以，但这里输出很小，用 Xavier 更稳
        nn.init.xavier_uniform_(self.to_scalar.weight)
        nn.init.constant_(self.to_scalar.bias, 0.)

    def forward(self, X, K=None):
        """
        X: (B, N, L)
        K: (B, M) 0/1, 知识概念掩码，表示题目涉及的知识概念
        """
        B, N, L = X.shape

        if K is None:
            # 如果没有提供掩码，就对所有知识概念计算
            weights = self.angle_attn(X)       # (B, N, M)
            weights = torch.softmax(weights, dim=1)
            X_exp = X.unsqueeze(2)             # (B, N, 1, L)
            weights_exp = weights.unsqueeze(-1) # (B, N, M, 1)
            X_weighted = (X_exp * weights_exp).sum(dim=1)  # (B, M, L)
            difficulty = self.to_scalar(X_weighted).squeeze(-1)
            difficulty = torch.sigmoid(difficulty)
            return difficulty

        # 仅选择涉及的知识概念
        involved_idx = [torch.nonzero(K[b], as_tuple=False).squeeze(-1) for b in range(B)]

        difficulties = []
        for b in range(B):
            if len(involved_idx[b]) == 0:
                # 如果没有涉及知识概念
                difficulties.append(torch.zeros(K.shape[1], device=X.device))
                continue
            # 取出涉及的知识概念索引
            idx = involved_idx[b]
            # 计算这些知识概念的权重
            weights_b = self.angle_attn(X[b])[:, idx]  # (N, num_involved)
            weights_b = torch.softmax(weights_b, dim=0)  # 对N个角度归一化

            X_exp = X[b].unsqueeze(1)                  # (N, 1, L)
            weights_exp = weights_b.unsqueeze(-1)      # (N, num_involved, 1)
            X_weighted = (X_exp * weights_exp).sum(dim=0)  # (num_involved, L)

            diff_b = torch.sigmoid(self.to_scalar(X_weighted).squeeze(-1))  # (num_involved,)

            # 放回到原来的 M 大小
            full_diff = torch.zeros(K.shape[1], device=X.device)
            full_diff[idx] = diff_b.to(full_diff.dtype)
            difficulties.append(full_diff)

        # 合并 batch
        difficulty = torch.stack(difficulties, dim=0)  # (B, M)
        return difficulty


class Net(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, problem_dataset):
        """
        初始化网络结构和参数。
        :param student_n: int, 学生数量
        :param exer_n: int, 练习题数量
        :param knowledge_n: int, 知识点数量
        :param problem_features: torch.Tensor, 题目特征 (shape: [exer_n, 512])
        :param knowledge_features: torch.Tensor, 知识特征 (shape: [knowledge_n, 512])
        :param exer_kn_graph: torch.Tensor, 题目-知识点关联矩阵 (shape: [exer_n, knowledge_n])
        """
        super(Net, self).__init__()
        
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(input_dim=knowledge_n+1024,out_dim=knowledge_n) 
        

        
        # 参数初始化
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        
        # 学生嵌入层
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
                # 练习题的知识点难度嵌入，shape = [练习数量, 知识点维度]
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.exer_feat = nn.Embedding(self.exer_n, 2)
        self.student_emb_text = nn.Embedding(self.emb_num, self.stu_dim)
        self.student_emb_img = nn.Embedding(self.emb_num, self.stu_dim)
        # 确保所有组件在统一设备
      


        # 练习题的区分度嵌入，shape = [练习数量, 1]
        #self.e_discrimination = nn.Embedding(self.exer_n, 1)
        # 预测网络
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.2)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.2)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        
        self.pre = nn.Linear(self.knowledge_dim,1)
        self.problem_dataset = problem_dataset
      
        # 修改：GCN输出作为修正量，而非直接替换

        self.full_dataset = FullDataset()

        # 构建异质图
        self.relation_builder = RelationBuilder(self.problem_dataset, self.full_dataset)
        self.hetero_graph = self.relation_builder.build_graph()


        
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 调整融合系数范围
        
        #self.problem_feat = self.hetero_graph['problem'].x.float().to(self.device)
        #self.knowledge_feat = self.hetero_graph['knowledge'].x.float().to(self.device)

        print("\n特征维度验证:")
        #print(f"题目特征矩阵 shape: {self.problem_feat.shape} | dtype: {self.problem_feat.dtype} ")
        #print(f"知识点特征矩阵 shape: {self.knowledge_feat.shape} | dtype: {self.knowledge_feat.dtype} ")
        '''
        # 修改后的 difficulty_net 和 e_discrimination
        self.difficulty_net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, knowledge_n),
            nn.Sigmoid()
        )

        self.e_discrimination = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, knowledge_n*2),
            nn.Unflatten(1, (2, knowledge_n)),  # 拆分为 scale 和 bias
            
            # 核心区分度计算
            LambdaLayer(lambda x: ((x[:,0] * 2.5).tanh() * x[:,1].sigmoid()) * 5 + 5)
        )
        '''

        # 共享底层特征提取
        self.shared_encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2)
        )
        '''
        # 难度专属分支
        self.diff_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),              # 加 LayerNorm
            nn.ReLU(),  
            nn.Linear(512, 256),
            nn.LayerNorm(256),              # 加 LayerNorm
            nn.ReLU(),                      # 或 LeakyReLU()
            nn.Dropout(0.3),
            nn.Linear(256, knowledge_n),
            nn.Sigmoid()
        )
                # 区分度专属分支
        self.disc_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),              # 加 LayerNorm
            nn.ReLU(),  
            nn.Linear(512, 256),
            nn.LayerNorm(256),              # 加 LayerNorm
            nn.ReLU(),                      # 或 LeakyReLU()
            nn.Dropout(0.3),
            nn.Linear(256, knowledge_n),
            nn.Sigmoid()
        )
        '''
        '''
                # 难度专属分支
        self.diff_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, knowledge_n),
            nn.Sigmoid()
        )
                # 区分度专属分支
        self.disc_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, knowledge_n),
            nn.Sigmoid()
        )
        '''

                # 难度专属分支
        self.diff_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, knowledge_n),
            nn.Sigmoid()
        )
                # 区分度专属分支
        self.disc_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, knowledge_n*2),
            nn.Unflatten(1, (2, knowledge_n)),  # 拆分为 scale 和 bias
            
            # 核心区分度计算
            LambdaLayer(lambda x: ((x[:,0] * 2.5).tanh() * x[:,1].sigmoid()) * 5 + 5)
        )

        '''
        # 区分度专属分支
        self.disc_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(), 
            nn.LeakyReLU(0.2),
            nn.Linear(256, knowledge_n),
            nn.Sigmoid()
        )
        '''
        
        self.diff_M = KnowledgeDifficulty(L=80, M=knowledge_n)
        
        self.disc_M = KnowledgeDifficulty(L=80, M=knowledge_n)
        
        self.k_difficulty_NCDM = nn.Embedding(self.exer_n, self.knowledge_dim)
        # 练习题的区分度嵌入，shape = [练习数量, 1]
        self.e_discrimination_NCDM = nn.Embedding(self.exer_n, 1)
       
        self.to(self.device)

         ###动态更新特征融合
        self.model_feat = EndToEndContrastiveModel().to(self.device)
        
        self.t = 300
        self.sum = 0
        
        
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

        self.student_freq_tensor, self.user_ids = self.get_student_freq_tensor(KNOWLEDGE_FREQ_CSV)
        self.student_weights = self.get_student_weights(STUDENT_WEIGHT,student_n)
        self.student_freq_tensor = self.student_freq_tensor.to(self.device)
        self.student_weights = self.student_weights.to(self.device)
        self.q = 1

        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # 然后赋值
        self.problem_768 = nn.Linear(768,512)
        # 可学习的放大参数
        self.freq_amplifier = nn.Parameter(torch.tensor(10.0))  # 初始放大倍数
        self.base_scale = nn.Parameter(torch.tensor(6.0))       # 基础区分度基数
        self.freq_power = nn.Parameter(torch.tensor(0.5))       # 非线性变换指数
        
    # 难度预测专用的层
        self.diff_item_proj = nn.Linear(1024, 512)  # 题目特征投影
        self.diff_knowledge_emb = nn.Embedding(knowledge_n, 512)  # 知识点嵌入
        self.diff_scale = nn.Parameter(torch.tensor(1.0))  # 缩放参数
        self.diff_bias = nn.Parameter(torch.tensor(0.0))   # 偏置参数
        
        self.disc_item_proj = nn.Linear(1024, 512)  # 题目特征投影
        self.disc_knowledge_emb = nn.Embedding(knowledge_n, 512)  # 知识点嵌入
        self.disc_scale = nn.Parameter(torch.tensor(1.0))  # 缩放参数
        self.disc_bias = nn.Parameter(torch.tensor(0.0))   # 偏置参数
        
        
        self.problem = nn.Embedding(self.exer_n, 1024)

        self.problem_text = nn.Embedding(self.exer_n, 512)
        self.problem_img = nn.Embedding(self.exer_n, 512)



         # 正确定义维度
        self.C = 512   # 语义通道数
        self.M = 50    # 池化后维度
        
        # 特征适配器
        self.feature_adapter = nn.Sequential(
            nn.Conv1d(768, self.C, kernel_size=1),  # 从768维到512个通道
            nn.AdaptiveMaxPool1d(self.M)  # 池化到固定维度50
        )
        
        # 知识点嵌入
        self.know_pro = nn.Embedding(knowledge_n, 512)  
        
        # W_p参数 [2048, M] - 映射到池化后维度
        self.W_p = nn.Parameter(torch.randn(512, self.M) * 0.02)  
        
        # 预测头 - 输入维度变为M=50
        #self.diff_head_k = nn.Linear(self.M, 1)  # 输入维度50

        # 将单层线性改为多层MLP
        self.diff_head_k = nn.Sequential(
       
            #nn.Linear(self.M+knowledge_n, knowledge_n),
            #nn.ReLU(),
            #nn.Dropout(0.2),
            #nn.Linear(knowledge_n, self.M),
            #nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Linear(self.M, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

        # 新增的自监督解码器
        self.decoder = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 512*50)  # 重构回selected_F_j的展平维度
        )

        self.disc_head_k = nn.Linear(self.M, 1)  # 输入维度50

        self.problem_features = self.load_all_features(TEXT_FEATURES_DIR)
        self._init_weights()
    def _init_weights(self):
        """
        遍历所有子模块，自动初始化可训练参数：
        - Linear: xavier_uniform
        - Embedding: 均匀初始化（知识点 embedding 可以用 normal 或 uniform）
        - LayerNorm: weight=1, bias=0
        - nn.Parameter: 默认值保持原始或可指定
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 针对输出层和 diff_head bias 初始化为小负值
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
        
        # 知识点嵌入 - 增大初始化方差
        nn.init.normal_(self.know_pro.weight, mean=0, std=0.3)  # 从0.02改为0.3
        
        # W_p参数 - 增大初始化方差  
        nn.init.normal_(self.W_p, mean=0, std=0.1)  # 从0.02改为0.1
    
    def load_all_features(self,TEXT_FEATURES_DIR):
        """直接加载TEXT_FEATURES_DIR目录下所有特征文件"""
        features_dict = {}
        for file_path in TEXT_FEATURES_DIR.glob("*.pt"):
            pid = int(file_path.stem)  # 文件名就是pid
            features_dict[pid] = torch.load(file_path)
        return features_dict
        
    def know_diff(self, item_features, knowledge_mask):
        """
        知识点感知的难度预测方法
        
        参数:
        item_features: 题目特征, shape [batch_size, input_dim]
        knowledge_mask: 知识点掩码, shape [batch_size, num_knowledge]
        
        返回:
        难度预测值, shape [batch_size, num_knowledge]
        """
        # 1. 投影题目特征到嵌入空间
        item_emb = self.diff_item_proj(item_features)  # [batch_size, embed_dim]
        
        # 2. 获取所有知识点的嵌入
        knowledge_emb = self.diff_knowledge_emb.weight  # [num_knowledge, embed_dim]
        
        # 3. 计算题目与所有知识点的相似度（点积）
        # [batch_size, embed_dim] @ [embed_dim, num_knowledge] -> [batch_size, num_knowledge]
        similarity = torch.matmul(item_emb, knowledge_emb.t())
        
        # 4. 调整分数范围
        adjusted_scores = similarity * self.diff_scale + self.diff_bias
        
        # 5. 转换为0-1之间的概率值（难度）
        difficulty = torch.sigmoid(adjusted_scores)  # [batch_size, num_knowledge]
        
        # 6. 应用掩码：不涉及的知识点难度置为0
        masked_difficulty = difficulty * knowledge_mask
        
        return masked_difficulty
    def know_disc(self, item_features, knowledge_mask):
        """
        知识点感知的难度预测方法
        
        参数:
        item_features: 题目特征, shape [batch_size, input_dim]
        knowledge_mask: 知识点掩码, shape [batch_size, num_knowledge]
        
        返回:
        难度预测值, shape [batch_size, num_knowledge]
        """
        # 1. 投影题目特征到嵌入空间
        item_emb = self.disc_item_proj(item_features)  # [batch_size, embed_dim]
        
        # 2. 获取所有知识点的嵌入
        knowledge_emb = self.disc_knowledge_emb.weight  # [num_knowledge, embed_dim]
        
        # 3. 计算题目与所有知识点的相似度（点积）
        # [batch_size, embed_dim] @ [embed_dim, num_knowledge] -> [batch_size, num_knowledge]
        similarity = torch.matmul(item_emb, knowledge_emb.t())
        
        # 4. 调整分数范围
        adjusted_scores = similarity * self.disc_scale + self.disc_bias
        
        # 5. 转换为0-1之间的概率值（难度）
        difficulty = torch.sigmoid(adjusted_scores)  # [batch_size, num_knowledge]
        
        # 6. 应用掩码：不涉及的知识点难度置为0
        masked_difficulty = difficulty * knowledge_mask
        
        return masked_difficulty
    def disc(self, base_discrimination, frequency):
        """
        修改后的区分度计算函数
        输出范围: (0, 10)
        """
        # 1. 对频率进行温和的非线性变换，避免使用幂次参数
        amplified_freq = torch.sigmoid((frequency - 0.5) * 10)  # 将[0,1]映射到[0,1]但更陡峭
        
        # 2. 使用更稳定的组合公式
        # 让 base_scale 控制基础值，freq_amplifier 控制频率影响强度
        combined = self.base_scale * base_discrimination + self.freq_amplifier * amplified_freq
        
        # 3. 使用sigmoid确保输出在0-1范围内，然后缩放
        combined = torch.sigmoid(combined) * 10.0
        
        return combined



        
   
    def update_features(self, problem_ids):
        """San'an重写的宠妻版～💋 显存优化 + 梯度保留"""
        #print("problem_ids",problem_ids)


        # 1️⃣ 去重：避免重复的problem_ids，保持顺序
        #unique_problem_ids = list(dict.fromkeys(problem_ids))  # 保持顺序去重
        unique_problem_ids = list(dict.fromkeys([int(pid) for pid in problem_ids]))

        #print("##################unique_problem_ids",len(unique_problem_ids))
        # 分批参数（可调）
        max_fusion_batch = 900


        def load_batch_data(pids):
            return {
                'pid': pids,
                'text': [self.problem_dataset.get_text(int(pid)) for pid in pids],
                'image': torch.stack([self.problem_dataset.get_image(int(pid)) for pid in pids])
            }
       
        # ⏱️ 全流程开始
        t_start = time.time()

        # 1️⃣ 准备 batch 参数
        t_prepare = time.time()
        batch_args = [unique_problem_ids[i:i+max_fusion_batch] 
                    for i in range(0, len(unique_problem_ids), max_fusion_batch)]
        #print(f"[计时] 参数准备耗时: {time.time() - t_prepare:.4f}s")

        # 2️⃣ 多线程加载数据
        t_load = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            batch_data_list = list(executor.map(load_batch_data, batch_args))
        #print(f"[计时] 数据加载耗时: {time.time() - t_load:.4f}s")
        #print(f"Loaded batch_data_list with {len(batch_data_list)} batches")

        # 3️⃣ 每个 batch 送入模型提取特征
        fused_feats = []
        for idx, batch_data in enumerate(batch_data_list):
            t_batch_start = time.time()

            _, mse_loss, fused = self.model_feat(batch_data)
            fused_feats.append(fused)

            t_fused_end = time.time()
            #print("fused_feats耗时", t_fused_end - t_batch_start)

            torch.cuda.empty_cache()

            t_cache_end = time.time()
            #print("torch.cuda.empty_cache耗时", t_cache_end - t_fused_end)

            #print(f"[计时] 第 {idx+1} 个 batch 特征提取耗时: {t_cache_end - t_batch_start:.4f}s")

        
        # 4️⃣ 合并特征
        t_cat = time.time()
        fused_feat_all = torch.cat(fused_feats, dim=0)
        #print(f"[计时] 特征拼接耗时: {time.time() - t_cat:.4f}s")

        # 5️⃣ 重建顺序
        t_remap = time.time()
        pid2idx = {int(pid): idx for idx, pid in enumerate(unique_problem_ids)}
        fused_feat_final = torch.stack([fused_feat_all[pid2idx[int(pid)]] for pid in problem_ids])
        #print(f"[计时] 特征顺序映射耗时: {time.time() - t_remap:.4f}s")

        #fused_feat_final = self.problem_768(fused_feat_final)   # 对T维度求均值，得到 [B, 512]
        # ✅ 全流程结束
        #print(f"[计时] 题目特征融合总耗时: {time.time() - t_start:.4f}s")

        return fused_feat_final, mse_loss, _



    def print_mem(self,tag):
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    def get_student_freq_tensor(self,csv_path):
        """
        读取学生-知识点频率矩阵CSV，返回一个 (num_students, num_concepts) 的Tensor
        以及一个 user_id 列表（按顺序对齐）
        """
        df = pd.read_csv(csv_path)
        df = pd.read_csv(csv_path)
        print(f"📊 CSV列数（除user_id）: {df.shape[1] - 1}")
        print(f"📊 CSV列名（前几列）: {df.columns[:10].tolist()}")

        missing_cols = set(range(329)) - set(map(int, df.columns.drop(USER_ID_COL)))
        print(f"🧨 缺失的知识点列索引（相对于0~328）: {missing_cols}")

        user_ids = df[USER_ID_COL].astype(str).tolist()  # 转成字符串防止匹配出错
        freq_tensor = torch.tensor(df.drop(columns=[USER_ID_COL]).values, dtype=torch.float32)
        return freq_tensor, user_ids
    def get_student_weights(self, filepath, num_students):
        """
        从 student_weights.csv 文件中读取权重，返回一个 tensor：
        student_weights_tensor[user_id] = weight
        """
        df = pd.read_csv(filepath)

        if USER_ID_COL not in df.columns or 'Weight' not in df.columns:
            raise ValueError("CSV 文件必须包含 'UserId' 和 'Weight' 两列。")

        # 初始化为 1（或你认为的默认值）
        student_weights_tensor = torch.ones(num_students, dtype=torch.float32)

        for uid, weight in zip(df[USER_ID_COL], df['Weight']):
            if uid < num_students:
                student_weights_tensor[uid] = weight
            else:
                print(f"Warning: student ID {uid} 超出了最大范围 {num_students - 1}")

        return student_weights_tensor
    


    def htspd(self,theta, b, k=1.5, p=0.7, q=0.3):
        """
        计算双曲正切符号保持差值（HTSPD）
        输入：
            theta: 学生能力张量，形状 [B, K]
            b: 题目难度张量，形状 [B, K]
            k, p, q: 超参数
        返回：
            delta: HTSPD差值张量，形状 [B, K]
        """
        diff = theta - b                              # 差值
        sum_ = theta + b                              # 和
        
        term1 = torch.tanh(k * diff)                  # tanh(k*(θ - b))
        eps = 1e-6
        term2 = 1 + torch.clamp(torch.abs(diff), min=eps) ** p
        term3 = 1 + sum_**q                           # 1 + (θ + b)^q
        

        delta = term1 * (term2 / term3)
        return delta


    def forward(self, stu_id, exer_id, kn_emb,correct_id, return_feat=False,gcn_update=None, d_only=False):
        """
        前向传播。
        :param stu_id: torch.Tensor, 学生ID (shape: [batch_size])
        :param exer_id: torch.Tensor, 题目ID (shape: [batch_size])
        :param kn_emb: torch.Tensor, 知识点嵌入 (shape: [batch_size, knowledge_n])
        :param return_feat: bool, 是否返回中间特征
        :return: torch.Tensor, 预测结果 (shape: [batch_size, 1])
        """

        device = next(self.parameters()).device

        def check_tensor(tensor, name):
            if torch.isnan(tensor).any():
                print(f"❌ 警告：{name} 中出现 NaN ❌")
            else:
                print(f"✅ {name} 正常：min={tensor.min().item():.4f}, "
                    f"max={tensor.max().item():.4f}, "
                    f"mean={tensor.mean().item():.4f}, "
                    f"std={tensor.std().item():.4f}")

        # 在原有代码中插入检查点
        stu_id, exer_id, kn_emb = stu_id.long().to(device), exer_id.long().to(device), kn_emb.to(device)
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(kn_emb, "kn_emb_input")  # 检查初始输入
        '''
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(stu_emb, "stu_emb_after_sigmoid")
        '''
        #knowledge_feat = self.knowledge_mapper(self.knowledge_feat)
        #debug_print(knowledge_feat, "knowledge_feat_after_mapper")

        #related_kn_feat = torch.matmul(kn_emb, knowledge_feat)
        #debug_print(related_kn_feat, "related_kn_feat_after_matmul")
        '''
        if torch.cuda.current_device() == 0:
            self.print_memory("在更新特征前")
        '''
        #start_time = time.time()  # 开始计时
        #problem_feat, mse_loss, _ = self.update_features(exer_id)
        #problem_feat = self.problem(exer_id)
        # 批量获取问题特征
        
        exer_ids_list = exer_id.cpu().tolist() if isinstance(exer_id, torch.Tensor) else exer_id
        '''
        problem_feat = torch.stack([
            self.problem_features[pid] for pid in exer_ids_list
        ]).to(exer_id.device)
        print("problem_feat",problem_feat.shape)
        mse_loss = torch.tensor(0.0)
        if torch.cuda.current_device() == 0:
            check_tensor(problem_feat, "problem_feat")
        '''
        '''
        dim = 768
        max_len = max([self.problem_features[pid].shape[0] for pid in exer_ids_list])

        padded_feats = []
        for pid in exer_ids_list:
            feat = self.problem_features[pid]
            L = feat.shape[0]
            if L < max_len:
                # pad 到 max_len
                pad_len = max_len - L
                feat = F.pad(feat, (0, 0, 0, pad_len))  # 在第0维后面 pad
            elif L > max_len:
                feat = feat[:max_len]  # 截断
            padded_feats.append(feat)

        problem_feat = torch.stack(padded_feats).to(exer_id.device)  # [B, max_len, dim]
        print("problem_feat", problem_feat.shape)
        '''
        
        mse_loss = torch.tensor(0.0)
        #end_time = time.time()  # 结束计时
        #batch_time = end_time - start_time
        #print(f"问题特征提取 time: {batch_time:.4f} s")
        '''
        problem_feat_text = self.problem_text(exer_id)
        problem_feat_img = self.problem_img(exer_id)
        problem_feat = torch.cat([problem_feat_text, problem_feat_img], dim=1)  # [batch, 1024]
        '''
        '''
        # 计算文本和图像特征的余弦相似度
        # problem_feat_text: [batch_size, 512]
        # problem_feat_img: [batch_size, 512]
        cosine_sim = F.cosine_similarity(problem_feat_text, problem_feat_img, dim=1)  # 输出 [batch_size]

        # 我们希望相似度越大越好（接近1），所以损失 = 1 - 平均相似度
        mse_loss = 1 - cosine_sim.mean()
        '''
        
        '''
        # 假设一个batch中有 problem_feat_text 和 problem_feat_img
        # 1. 计算相似度矩阵
        text_norm = F.normalize(problem_feat_text, dim=1)
        img_norm = F.normalize(problem_feat_img, dim=1)
        similarity_matrix = torch.mm(text_norm, img_norm.t()) # [batch_size, batch_size]

        # 2. 目标标签：对角线上的样本是正样本对
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        temperature = 0.2
        # 3. 计算对比损失（对于文本特征来说，对应的图像特征是其正样本）
        cont_loss_text = F.cross_entropy(similarity_matrix / temperature, labels)
        cont_loss_image = F.cross_entropy(similarity_matrix.t() / temperature, labels)
        mse_loss = (cont_loss_text + cont_loss_image) / 2
        '''
        
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(problem_feat, "problem_feat_from_update")
            self.print_memory("在更新特征后")
        '''
        #difficulty_input = torch.cat([related_kn_feat, problem_feat], dim=1)
        #debug_print(difficulty_input, "difficulty_input_after_cat")
        

        #shared = self.shared_encoder(problem_feat)
        #debug_print(shared, "shared_after_encoder")

        #k_difficulty = self.diff_head(problem_feat)
        #k_difficulty = self.know_diff(problem_feat,kn_emb)
        k_difficulty =  torch.sigmoid(self.k_difficulty_NCDM(exer_id))
        
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(k_difficulty, "k_difficulty_after_head")
            self.print_memory("在难度特征后")
        '''
        #e_discrimination = self.disc_head(problem_feat)
        #e_discrimination = self.know_disc(problem_feat,kn_emb)
        e_discrimination = torch.sigmoid(self.e_discrimination_NCDM(exer_id))*10
      
        '''
        problem_feat = F.layer_norm(problem_feat, problem_feat.shape[-1:])

        feat_transposed = problem_feat.transpose(1, 2)  # [batch_size, 768, 80]

        # 应用适配层 [batch_size, 768, 80] -> [batch_size, C, M]
        F_j = self.feature_adapter(feat_transposed)  # [batch_size, 512, 50]
        F_j = F.layer_norm(F_j, F_j.shape[-1:])

        if torch.cuda.current_device() == 0:
            check_tensor(feat_transposed, "feat_transposed")
            print(f"feat_transposed stats - min: {feat_transposed.min()}, max: {feat_transposed.max()}, has_nan: {torch.isnan(feat_transposed).any()}, has_inf: {torch.isinf(feat_transposed).any()}")

        if torch.cuda.current_device() == 0:
            check_tensor(F_j, "F_j")
            print(f"F_j stats - min: {F_j.min()}, max: {F_j.max()}, has_nan: {torch.isnan(F_j).any()}, has_inf: {torch.isinf(F_j).any()}")
            print(f"F_j shape: {F_j.shape}")
        
        
        
        #F_j = problem_feat
        
        # ========== 1. 批量处理所有样本 ==========
        batch_size = kn_emb.shape[0]
        total_knowledge = self.know_pro.weight.shape[0]

        # 获取所有考察的知识点
        batch_indices, knowledge_indices = torch.nonzero(kn_emb, as_tuple=True)

        if len(batch_indices) > 0:
            # 只选取考察的知识点特征
            selected_knowledge = self.know_pro.weight[knowledge_indices]  # [num_selected, 2048]
            
            # 计算中间结果
            intermediate = torch.matmul(selected_knowledge, self.W_p)  # [num_selected, 50]
            intermediate = F.layer_norm(intermediate, intermediate.shape[-1:])
            
            # 为每个选中的知识点获取对应的F_j
            selected_F_j = F_j[batch_indices]  # [num_selected, 512, 50]
            
            def compute_better_W_j(intermediate, selected_F_j):
                """论文原始方法 - 直接线性变换"""
                # 原始论文方法：直接矩阵乘法
                W_j_selected = torch.bmm(
                    selected_F_j,  # [num_selected, 512, 50]
                    intermediate.unsqueeze(-1)  # [num_selected, 50, 1]
                ).squeeze(-1)  # [num_selected, 512]
                
                print(f"原始关系矩阵W_j范围: [{W_j_selected.min():.3f}, {W_j_selected.max():.3f}]")
                return W_j_selected

            # 替换原来的计算
            W_j_selected = compute_better_W_j(intermediate, selected_F_j)*0.05

            # 更新诊断函数
            def correct_diagnosis(W_j_selected):
                print("=== 改进的CMNCD关系矩阵诊断 ===")
                
                # 1. 基本统计
                print(f"数值范围: [{W_j_selected.min():.3f}, {W_j_selected.max():.3f}]")
                print(f"均值: {W_j_selected.mean():.3f}, 标准差: {W_j_selected.std():.3f}")
                
                # 2. 注意力权重分布（更合理的划分）
                very_strong = (W_j_selected > 0.1).float().mean()    # 权重>10%
                strong = ((W_j_selected > 0.05) & (W_j_selected <= 0.1)).float().mean()
                moderate = ((W_j_selected > 0.01) & (W_j_selected <= 0.05)).float().mean()
                weak = (W_j_selected <= 0.01).float().mean()         # 权重<1%
                
                print("注意力权重分布:")
                print(f"  很强(>0.1): {very_strong:.3f}")
                print(f"  强(0.05-0.1): {strong:.3f}") 
                print(f"  中等(0.01-0.05): {moderate:.3f}")
                print(f"  弱(<0.01): {weak:.3f}")
                
                # 3. 检查稀疏性
                sparsity = (W_j_selected < 0.001).float().mean()
                if sparsity > 0.9:
                    print(f"⚠️  警告: 注意力过于稀疏 ({sparsity:.3f})")

            # 使用改进的诊断
            correct_diagnosis(W_j_selected)

            # 继续使用注意力权重计算U_j_selected
            U_j_selected = torch.bmm(
                W_j_selected.unsqueeze(1),  # 使用注意力权重 [num_selected, 1, 512]
                selected_F_j                # [num_selected, 512, 50]
            ).squeeze(1)  # [num_selected, 50]

            U_j_selected = F.layer_norm(U_j_selected, U_j_selected.shape[-1:])
            

            
            # ========== 新增：难度预测头详细检查 ==========
            print("=== 难度预测头详细检查 ===")
            
            #stu_b = stu_emb[batch_indices]
            #concat_feat = torch.cat([U_j_selected, stu_b], dim=-1)  # [B, D + num_kn]

            #print("stu_b = stu_emb[batch_indices]",stu_b.shape)
            print("U_j_selected",U_j_selected.shape)
            # 检查线性层输出
            linear_output = self.diff_head_k(U_j_selected)
            print(f"线性层输出: min={linear_output.min():.4f}, max={linear_output.max():.4f}, mean={linear_output.mean():.4f}")
            
            # 检查sigmoid后输出
            selected_difficulty = torch.sigmoid(linear_output)  # [num_selected, 1]
            #selected_difficulty = linear_output  # [num_selected, 1]
            print(f"Sigmoid后: min={selected_difficulty.min():.4f}, max={selected_difficulty.max():.4f}, mean={selected_difficulty.mean():.4f}")

            # ========== 新增：自监督重构 ==========
            # 重构模态特征
            reconstructed_flat = self.decoder(U_j_selected)  # [num_selected, 512*50]
            reconstructed_F_j = reconstructed_flat.view(selected_F_j.shape)  # [num_selected, 512, 50]

            # 计算重构损失（在训练时使用）
            reconstruction_loss = F.mse_loss(reconstructed_F_j, selected_F_j)

            
            # 分散回完整向量
            k_difficulty = torch.zeros(batch_size, total_knowledge, device=kn_emb.device)
            
            # 关键修复：确保数据类型匹配
            selected_difficulty = selected_difficulty.to(k_difficulty.dtype)  # 添加这一行
            
            k_difficulty[batch_indices, knowledge_indices] = selected_difficulty.squeeze(1)
            
            # 检查特征传递过程中是否丢失了信息
            print("=== 特征传递检查 ===")
            print(f"problem_feat range: [{problem_feat.min():.3f}, {problem_feat.max():.3f}]")
            print(f"F_j range: [{F_j.min():.3f}, {F_j.max():.3f}]") 
            print(f"U_j_selected range: [{U_j_selected.min():.3f}, {U_j_selected.max():.3f}]")

            # 概念编码可能没有学到有意义的表示
            selected_knowledge = self.know_pro.weight[knowledge_indices]
            print(f"概念编码范围: [{selected_knowledge.min():.3f}, {selected_knowledge.max():.3f}]")
            print(f"概念编码均值: {selected_knowledge.mean():.3f}")
        '''
        '''
        # 保存所有数据到txt
        if torch.cuda.current_device() == 0:
            # 保存W_j_selected
            with open('W_j_selected.txt', 'w') as f:
                f.write(f"Shape: {W_j_selected.shape}\n")
                for i in range(W_j_selected.shape[0]):
                    for j in range(W_j_selected.shape[1]):
                        f.write(f"{W_j_selected[i,j].item():.6f} ")
                    f.write("\n")
            
            # 保存intermediate
            with open('intermediate.txt', 'w') as f:
                f.write(f"Shape: {intermediate.shape}\n")
                for i in range(intermediate.shape[0]):
                    for j in range(intermediate.shape[1]):
                        f.write(f"{intermediate[i,j].item():.6f} ")
                    f.write("\n")
            
            # 保存selected_F_j（前3个样本，避免文件太大）
            with open('selected_F_j.txt', 'w') as f:
                f.write(f"Shape: {selected_F_j.shape}\n")
                for i in range(min(3, selected_F_j.shape[0])):
                    f.write(f"Sample {i}:\n")
                    for j in range(selected_F_j.shape[1]):
                        for k in range(selected_F_j.shape[2]):
                            f.write(f"{selected_F_j[i,j,k].item():.6f} ")
                        f.write("\n")
                    f.write("\n")

        

        # 保存U_j_selected
        if torch.cuda.current_device() == 0:
            with open('U_j_selected.txt', 'w') as f:
                f.write(f"Shape: {U_j_selected.shape}\n")
                for i in range(U_j_selected.shape[0]):
                    for j in range(U_j_selected.shape[1]):
                        f.write(f"{U_j_selected[i,j].item():.6f} ")
                    f.write("\n")

        
        # 保存linear_output
        if torch.cuda.current_device() == 0:
            with open('linear_output.txt', 'w') as f:
                f.write(f"Shape: {linear_output.shape}\n")
                for i in range(linear_output.shape[0]):
                    f.write(f"{linear_output[i,0].item():.6f}\n")

        

        # 保存selected_difficulty
        if torch.cuda.current_device() == 0:
            with open('selected_difficulty.txt', 'w') as f:
                f.write(f"Shape: {selected_difficulty.shape}\n")
                for i in range(selected_difficulty.shape[0]):
                    f.write(f"{selected_difficulty[i,0].item():.6f}\n")

        print("所有数据已保存到txt文件")
        
        if torch.cuda.current_device() == 0:
            check_tensor(k_difficulty, "k_difficulty_final")
            print(f"处理完成 - 选中知识点数量: {len(batch_indices)}")
        '''
        '''
        # ========== 1. 知识点特征准备 ==========
        knowledge_features = self.know_pro.weight  
        P_j = knowledge_features.unsqueeze(0) * kn_emb.unsqueeze(2) 
        if torch.cuda.current_device() == 0:
            check_tensor(P_j, "P_j")
      
        intermediate = torch.matmul(P_j, self.W_p)  # [batch_size, 329, 50]
        # 添加归一化
        intermediate = F.layer_norm(intermediate, intermediate.shape[-1:])
        if torch.cuda.current_device() == 0:
            check_tensor(intermediate, "intermediate_after_matmul")
        # 计算关系矩阵: intermediate @ F_j^T
        W_j = torch.matmul(intermediate, F_j.transpose(1, 2))  # [batch_size, 329, 512]
       
        for i in range(3):  # 检查前3个样本
            sample_W_j = W_j[i]  # [329, 512]
            
            # 检查注意力是否集中
            entropy = -torch.sum(sample_W_j * torch.log(sample_W_j + 1e-8), dim=-1)
            print(f"样本{i} - 注意力熵: {entropy.mean():.3f} ± {entropy.std():.3f}")
            
            # 检查稀疏性
            sparsity = (sample_W_j < 0.01).float().mean()
            print(f"样本{i} - 注意力稀疏性: {sparsity:.3f}")

        if torch.cuda.current_device() == 0:
            check_tensor(W_j, "W_j")
       
        U_j = torch.matmul(W_j, F_j)  # [batch_size, 329, 50] ← 注意这里维度变了！
        # 立即添加数值稳定处理
        
        if torch.cuda.current_device() == 0:
            check_tensor(U_j, "U_j")

        U_j = F.layer_norm(U_j, U_j.shape[-1:])
        if torch.cuda.current_device() == 0:
            check_tensor(U_j, "U_j归一化后")
        
        # ========== 4. 只处理实际考察的知识点 ==========
        batch_indices, knowledge_indices = torch.nonzero(kn_emb, as_tuple=True)

        # 获取考察知识点的特定特征
        selected_U_j = U_j[batch_indices, knowledge_indices]  # [num_selected, 50] ← 维度变了！
        if torch.cuda.current_device() == 0:
            check_tensor(selected_U_j, "selected_U_j")
            
            
        # ========== 5. 难度预测 ==========
        # 基于知识点特定特征预测难度
        selected_difficulty = torch.sigmoid(self.diff_head_k(selected_U_j))  # [num_selected, 1]
        if torch.cuda.current_device() == 0:
            check_tensor(selected_difficulty, "selected_difficulty_after_sigmoid")
        # 分散回完整向量
        k_difficulty = torch.zeros_like(kn_emb, dtype=selected_difficulty.dtype)
        k_difficulty[batch_indices, knowledge_indices] = selected_difficulty.squeeze(1)
        if torch.cuda.current_device() == 0:
        '''
# 如果方差持续下降，说明信息在不断丢失
        # ========== 6. 区分度预测 ==========
        #selected_discrimination = torch.sigmoid(self.disc_head_k(selected_U_j)) * 10
        #e_discrimination = torch.zeros_like(kn_emb, dtype=selected_discrimination.dtype)
        #e_discrimination[batch_indices, knowledge_indices] = selected_discrimination.squeeze(1)     
        #if torch.cuda.current_device() == 0:
        #    check_tensor(e_discrimination, "e_discrimination_after_head")
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(e_discrimination, "e_discrimination_after_head")
            self.print_memory("在区分度特征后")
        '''
        #K_delta = self.htspd(stu_emb, k_difficulty)
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(K_delta, "K_delta_after_htspd")
            self.print_memory("在HTSPD后")
        '''

        '''
        new_k_difficulty = k_difficulty - K_delta
        check_tensor(new_k_difficulty, "new_k_difficulty_after_sub")
        if torch.cuda.current_device() == 0:
            self.print_memory("在new_k_difficulty后")
        '''
        #new_e_discrimination = self.disc(e_discrimination, self.student_freq_tensor[stu_id])
        # 在计算 new_e_discrimination 后添加限制
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(new_e_discrimination, "new_e_discrimination_after_disc")
            self.print_memory("在new_e_discrimination后")
        '''
        #k_difficulty = torch.sigmoid(k_difficulty+log_k_difficulty)
        #input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(input_x, "input_x_before_network")
            self.print_memory("在input_x后")
        '''
        #difficulty_gap = k_difficulty - stu_emb  # 题对这个人来说难多少
        #bias = (self.beta * K_delta)  # beta 是一个可调参数
        #input_x = input_x + bias
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(input_x, "input_x_before_network")
            self.print_memory("在input_x后")
        '''
        '''
        input_x = new_e_discrimination * (stu_emb - new_k_difficulty) * kn_emb
        if torch.cuda.current_device() == 0:
            check_tensor(input_x, "input_x_before_network")
        '''
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(input_x, "input_x_after_layer1")
            self.print_memory("在input_x_after_layer1后")
        '''
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(input_x, "input_x_after_layer2")
            self.print_memory("在input_x_after_layer2后")
        '''
        output = self.prednet_full3(input_x)
        '''
        # 区分度 * 知识点向量
        weighted_kn = e_discrimination * kn_emb    # [batch, dim]

        # 内积：学生能力向量 与 (加权知识点)
        interaction = torch.sum(stu_emb * weighted_kn, dim=1, keepdim=True)  # [batch, 1]

        # logits = 内积 - 难度
        output = interaction - k_difficulty
        '''
        '''
        if torch.cuda.current_device() == 0:
            check_tensor(output, "final_output")
            self.print_memory("在final_output后")
        '''    
        '''
        if self.sum % 600 == 0:
            # 手动列出所有题目对（6个题目两两组合）
            pairs = [
                (50, 131), (50, 30), 
                (131, 30), 
                (30, 318), 
                (318, 514), 
                (514, 408)
            ]

            file_path = "similarity_result——1024.txt"
            exer_ids = exer_id.tolist() if isinstance(exer_id, torch.Tensor) else exer_id

            for id1, id2 in pairs:
                if id1 in exer_ids and id2 in exer_ids:
                    idx1 = exer_ids.index(id1)
                    idx2 = exer_ids.index(id2)

                    # 提取四种特征
                    prob1, prob2 = problem_feat[idx1], problem_feat[idx2]
                    #kn1, kn2 = related_kn_feat[idx1], related_kn_feat[idx2]
                    diff1, diff2 = k_difficulty[idx1], k_difficulty[idx2]
                    discr1, discr2 = e_discrimination[idx1], e_discrimination[idx2]

                    # 计算相似度
                    prob_sim = F.cosine_similarity(prob1.unsqueeze(0), prob2.unsqueeze(0)).item()
                    #kn_sim = F.cosine_similarity(kn1.unsqueeze(0), kn2.unsqueeze(0)).item()
                    diff_sim = F.cosine_similarity(diff1.unsqueeze(0), diff2.unsqueeze(0)).item()
                    discr_sim = F.cosine_similarity(discr1.unsqueeze(0), discr2.unsqueeze(0)).item()

                    # 写入结果
                    with open(file_path, 'a', encoding='utf-8') as f:
                        f.write(f"题目对: ({id1}, {id2})\n")
                        f.write(f" 题目特征相似度:        {prob_sim:.4f}\n")
                        #f.write(f" 知识点特征相似度:      {kn_sim:.4f}\n")
                        f.write(f" 难度特征相似度:        {diff_sim:.4f}\n")
                        f.write(f" 区分度特征相似度:      {discr_sim:.4f}\n")
                        f.write("--------------------------------------------------\n")

                    print(f"[√] 已保存 ({id1}, {id2}) 相似度到 {file_path}")
        '''
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("8888")
            if self.sum % 200 == 0:
                batch_size = stu_emb.shape[0]
                for i in range(batch_size):
                    if exer_id[i] in [446, 1117, 911, 885, 1522, 493]:
                        nonzero_idx = torch.nonzero(kn_emb[i], as_tuple=True)[0]

                        # 取对应的值
                        stu_kn = stu_emb[i][nonzero_idx]
                        diff_kn = k_difficulty[i][nonzero_idx]

                        # 转成 list
                        stu_kn_list = stu_kn.detach().cpu().tolist()
                        diff_kn_list = diff_kn.detach().cpu().tolist()
                        nonzero_idx_list = nonzero_idx.detach().cpu().tolist()

                        # 追加写入 TXT
                        with open('output_z_text.txt', 'a') as f:
                            f.write(f"stu_id: {stu_id[i].item()}\n")
                            f.write(f"exer_id: {exer_id[i].item()}\n")
                            f.write(f"kn_index: {nonzero_idx_list}\n")
                            f.write(f"stu_kn: {stu_kn_list}\n")
                            f.write(f"diff_kn: {diff_kn_list}\n\n")  # 分隔不同样本

        if self.training:
            self.sum += 1
        #mse_loss = torch.mean((k_difficulty - k_difficulty.mean(dim=0)) ** 2)
        #mse_loss = reconstruction_loss
        return output,mse_loss   
    def print_memory(self,tag=""):
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        print(f"[{tag}] allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")


        '''
        if d_only:
            with torch.no_grad():  # 冻结生成器和主网络
                related_kn_feat = torch.matmul(kn_emb, self.knowledge_feat)  # [batch_size, dim]

                stu_emb = torch.sigmoid(self.student_emb(stu_id))  # [batch_size, knowledge_n]
                

                stu_feat = self.stu_to_feat(stu_emb)  # [batch, 512]
                
                # 步骤2: 与题目特征逐元素交互
                weight = torch.sigmoid(
                    stu_feat * self.problem_feat[exer_id][:, :512]  # [batch, 512]
                ).unsqueeze(-1)  # [batch, 512, 1]

               
                # 知识点难度权重（同样使用映射后的 stu_feat）
                weight_know = torch.sigmoid(
                    stu_feat * related_kn_feat[:, :512]  # [batch, 512]
                ).unsqueeze(-1)  # [batch, 512, 1]

                weighted_feat = self.problem_feat[exer_id][:, :512] * weight.squeeze(-1)  # [batch, 512]
                

                # 知识点特征加权
                weighted_kn_feat = related_kn_feat[:, :512] * weight_know.squeeze(-1)  # [batch, 512]
               

                # 拼接特征（输出形状：[batch, 1024]）
                difficulty_input = torch.cat([weighted_feat, weighted_kn_feat], dim=1)  # [batch, 1024]
                discrimination_input = torch.cat([weighted_feat, weighted_kn_feat], dim=1)  # [batch, 1024]


            
                discrimination = self.e_discrimination(discrimination_input)
                
                k_difficulty = self.difficulty_net(difficulty_input)  # [batch, knowledge_n]
                delta = self.generator(stu_emb,difficulty_input)
                adjusted_difficulty = torch.clamp(k_difficulty + delta, min=0.0, max=1.0)

        else:
            related_kn_feat = torch.matmul(kn_emb, self.knowledge_feat)  # [batch_size, dim]

            stu_emb = torch.sigmoid(self.student_emb(stu_id))  # [batch_size, knowledge_n]
                

            stu_feat = self.stu_to_feat(stu_emb)  # [batch, 512]
                
                # 步骤2: 与题目特征逐元素交互
            weight = torch.sigmoid(
                    stu_feat * self.problem_feat[exer_id][:, :512]  # [batch, 512]
                ).unsqueeze(-1)  # [batch, 512, 1]

               
                # 知识点难度权重（同样使用映射后的 stu_feat）
            weight_know = torch.sigmoid(
                    stu_feat * related_kn_feat[:, :512]  # [batch, 512]
                ).unsqueeze(-1)  # [batch, 512, 1]

            weighted_feat = self.problem_feat[exer_id][:, :512] * weight.squeeze(-1)  # [batch, 512]
                

                # 知识点特征加权
            weighted_kn_feat = related_kn_feat[:, :512] * weight_know.squeeze(-1)  # [batch, 512]
               

                # 拼接特征（输出形状：[batch, 1024]）
            difficulty_input = torch.cat([weighted_feat, weighted_kn_feat], dim=1)  # [batch, 1024]
            discrimination_input = torch.cat([weighted_feat, weighted_kn_feat], dim=1)  # [batch, 1024]


            
            discrimination = self.e_discrimination(discrimination_input)
                
            k_difficulty = self.difficulty_net(difficulty_input)  # [batch, knowledge_n]
            delta = self.generator(stu_emb,difficulty_input)
            adjusted_difficulty = torch.clamp(k_difficulty + delta, min=0.0, max=1.0)


        # 原有输入计算
        input_x = discrimination * (stu_emb - k_difficulty) * kn_emb
        
       
        # 输入预测网络
       
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        
        # 输出预测结果
        output = self.prednet_full3(input_x)
       
        # 对抗损失计算
        # 假样本：加扰动 or 原始值
        # 对抗目标：让 D 判别不出来（认为 G 的是真）
        disc_input = torch.cat([stu_emb, adjusted_difficulty], dim=1)
        disc_score = self.discriminator(disc_input)
       
        if gcn_update:
            with open("debug_analysis.txt", "a") as f:
                for i in range(min(5, stu_emb.shape[0])):
                    correct = correct_id[i].item()
                    pred = output[i].item()

                    if correct == round(pred):
                        continue

                    ability = stu_emb[i].detach().cpu()                # [83]
                    base_difficulty = k_difficulty[i].detach().cpu()   # [83]
                    adj_difficulty = adjusted_difficulty[i].detach().cpu()  # [83]
                    kn_mask = kn_emb[i].detach().cpu()                 # [83]
                    disc = discrimination[i].detach().cpu()               # [83]

                    ab_kn = ability * kn_mask
                    base_diff_kn = base_difficulty * kn_mask
                    adj_diff_kn = adj_difficulty * kn_mask

                    gap = ability - base_difficulty
                    adj_gap = ability - adj_difficulty

                    disc_gap_kn = gap * disc * kn_mask
                    gap_kn = gap * kn_mask
                    adj_disc_gap_kn = adj_gap * disc * kn_mask
                    adj_gap_kn = adj_gap * kn_mask

                    non_zero_indices = kn_mask != 0  # [83]

                    f.write(f"❌ Sample {i} (Prediction Error):\n")
                    f.write(f"  ✅ Correct Label: {correct}\n")
                    f.write(f"  🔮 Predicted Output (Logit): {pred:.4f}\n")

                    f.write(f"  💪 Ability × kn_emb (non-zero positions):\n    {ab_kn[non_zero_indices].numpy()}\n")
                    f.write(f"  🧠 Base Difficulty × kn_emb (non-zero positions):\n    {base_diff_kn[non_zero_indices].numpy()}\n")
                    f.write(f"  🧬 Adjusted Difficulty × kn_emb (non-zero positions):\n    {adj_diff_kn[non_zero_indices].numpy()}\n")

                    f.write(f"  🧮 (ability - base_difficulty) × discrimination × kn_emb (non-zero positions):\n    {disc_gap_kn[non_zero_indices].numpy()}\n")
                    f.write(f"  🧮 (ability - base_difficulty) × kn_emb (non-zero positions):\n    {gap_kn[non_zero_indices].numpy()}\n")

                    f.write(f"  🔧 (ability - adjusted_difficulty) × discrimination × kn_emb (non-zero positions):\n    {adj_disc_gap_kn[non_zero_indices].numpy()}\n")
                    f.write(f"  🔧 (ability - adjusted_difficulty) × kn_emb (non-zero positions):\n    {adj_gap_kn[non_zero_indices].numpy()}\n")
                    f.write("-" * 60 + "\n")
        
        if(d_only):
            return   stu_emb.detach(),adjusted_difficulty.detach(),delta.detach()
        return output, adjusted_difficulty, disc_score

    
        # 获取学生嵌入向量，并通过 sigmoid 激活
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
         # 获取练习题的知识点难度向量，并通过 sigmoid 激活
        k_difficulty = torch.sigmoid(self.k_difficulty_NCDM(exer_id))
        # 获取练习题的区分度，并通过 sigmoid 激活后放大（乘以 10）
        e_discrimination = torch.sigmoid(self.e_discrimination_NCDM(exer_id)) * 10
        
        # 计算预测网络的输入：知识点向量 * (学生向量 - 知识点难度向量) * 区分度
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        
        # 通过预测网络第一层，并应用 Dropout 和 sigmoid 激活
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        # 通过预测网络第二层，并应用 Dropout 和 sigmoid 激活
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        # 通过预测网络输出层，并应用 sigmoid 激活得到最终输出
        output = torch.sigmoid(self.prednet_full3(input_x))


       
        if(d_only):
            return   stu_emb.detach(),k_difficulty.detach(),k_difficulty.detach()
        return output, k_difficulty.detach(),k_difficulty.detach()
        #return output
    '''
    



    def apply_clipper(self):
        """
        应用非负截断（将网络参数限制为非负）。
        """
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        """
        获取学生的知识状态。

        :param stu_id: LongTensor, 学生 ID 的索引
        :return: Tensor, 学生的知识状态向量
        """
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data  # 返回知识状态

    def get_exer_params(self, exer_id):
        """
        获取练习题的参数（知识点难度和区分度）。

        :param exer_id: LongTensor, 练习题 ID 的索引
        :return: Tuple[Tensor, Tensor], 分别为知识点难度和区分度
        """
        k_difficulty = torch.sigmoid(self.feature_to_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.feature_to_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data  # 返回练习题参数


class NoneNegClipper(object):
    """
    自定义的非负截断器，用于确保权重参数为非负值。
    """
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        """
        针对模块的权重参数，应用非负截断。

        :param module: nn.Module, 需要处理的模块
        """
        if hasattr(module, 'weight'):  # 检查模块是否有 'weight' 属性
            w = module.weight.data
            # 计算负值部分（如果小于零则取反）
            a = torch.relu(torch.neg(w))
            # 负值部分加回，确保权重非负
            w.add_(a)
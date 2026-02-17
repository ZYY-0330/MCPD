
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
class SNRDifficultyHead(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. 简单的聚焦层 (Focus): 把 196 维变成 1 维
        # 我们用知识点去"加权"图片，而不是生硬的 MaxPool
        self.attn_fc = nn.Linear(feature_dim, 1) 

        # 2. 难度预测器 (MLP)
        # 输入维度是 2: [相关强度(投影长), 干扰强度(噪声长)]
        # 或者我们可以输入更多几何特征，这里保持极简
        self.predictor = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, img_feat, know_feat):
        """
        img_feat:  [N, 196, 256] (N = Batch里这就道题涉及的知识点总数)
        know_feat: [N, 256]
        """
        
        # --- A. 聚焦 (Focus) ---
        # 既然要做向量减法，我们需要先把 196 个块合成一个"代表向量"
        # 这里用一种极简的 Softmax Attention
        # 逻辑：跟知识点越像的块，权重越大
        
        # [N, 196, 256] * [N, 1, 256] -> Sum(-1) -> [N, 196]
        scores = torch.sum(img_feat * know_feat.unsqueeze(1), dim=-1) 
        weights = F.softmax(scores, dim=1).unsqueeze(-1) # [N, 196, 1]
        
        # 加权求和: [N, 196, 256] * [N, 196, 1] -> Sum(1) -> [N, 256]
        focused_img = torch.sum(img_feat * weights, dim=1)
        
        # --- B. 几何分解 (Geometric Decomposition) ---
        
        # 1. 归一化 (关键！几何投影必须在单位球面上做)
        I_norm = F.normalize(focused_img, p=2, dim=-1)
        K_norm = F.normalize(know_feat, p=2, dim=-1)
        
        # 2. 计算投影 (有效信息 / Signal)
        # Dot Product: 这道题里有多少成分是属于这个知识点的？
        # [N, 1]
        relevance_scalar = torch.sum(I_norm * K_norm, dim=-1, keepdim=True)
        relevance_vec = relevance_scalar * K_norm
        
        # 3. 计算正交噪声 (干扰信息 / Noise)
        # 原始向量 - 有效向量 = 剩下的没用/干扰向量
        noise_vec = I_norm - relevance_vec
        
        # 计算噪声的模长 (Magnitude)
        noise_scalar = torch.norm(noise_vec, p=2, dim=-1, keepdim=True)
        
        # --- C. 预测难度 ---
        # 拼接 [有效性, 干扰性] -> [N, 2]
        # 神经网络会学会：有效性低 + 干扰性高 = 难
        geometric_features = torch.cat([relevance_scalar, noise_scalar], dim=-1)
        
        difficulty = torch.sigmoid(self.predictor(geometric_features))
        
        return difficulty
from torch.utils.checkpoint import checkpoint
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
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
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

        #self.student_freq_tensor, self.user_ids = self.get_student_freq_tensor(KNOWLEDGE_FREQ_CSV)
        #self.student_weights = self.get_student_weights(STUDENT_WEIGHT,student_n)
        #self.student_freq_tensor = self.student_freq_tensor.to(self.device)
        #self.student_weights = self.student_weights.to(self.device)
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
            nn.Conv1d(512, self.C, kernel_size=1),  # 从768维到512个通道
            nn.AdaptiveMaxPool1d(self.M)  # 池化到固定维度50
        )
        
        
        
       
        
        #self.problem_features = self.load_all_features(TEXT_FEATURES_DIR)



        self.output_layer = nn.Linear(knowledge_n, 1)
        self.feature_weights = nn.Parameter(torch.randn(329))  # 自定义329个权重
        '''
        csv_file = '/mnt/proj/autodl-tmp/data_2/XES3G5M/concept_frequency_percentage.csv'
        df = pd.read_csv(csv_file)

        self.vector_329 = np.zeros(329)  # 先创建全0向量
        for _, row in df.iterrows():
            cid = int(row['concept_id'])
            if 0 <= cid < 329:
                self.vector_329[cid] = row['percentage']

        # 现在 vector_329 就是长度为 329 的一维向量
        # vector_329[0] 对应 concept_id = 0
        # vector_329[100] 对应 concept_id = 100，依此类推

        print("向量长度:", len(self.vector_329))
        print("第0个知识点百分比:", self.vector_329[0])

        '''
        # 统一特征维度
        self.FEATURE_DIM = 256 # 768
        self.GATE_HIDDEN_DIM = 128     # 门控网络的隐藏层维度
        self.DIFF_HIDDEN_DIM = 128     # 难度头隐藏层维度
        self.LATENT_K_DIM = 256         # W_p 投影的知识点潜在维度 (你原先的40)

        # ----------------------------------------------------------------------
        # NCDM 基础层 (768维特征的融合起点)
        # ----------------------------------------------------------------------
        self.student_emb = nn.Embedding(student_n, self.knowledge_dim)
        self.k_difficulty_NCDM = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination_NCDM = nn.Embedding(self.exer_n, 1)

        # ----------------------------------------------------------------------
        # Fusion 核心层 (需适应 768 维)
        # ----------------------------------------------------------------------
        
        # W_p参数：必须从特征维度 (768) 投影到你的潜在维度 (40)
        self.W_p = nn.Parameter(torch.randn(self.FEATURE_DIM, self.LATENT_K_DIM) * 0.02)
        
        # 难度头 (输入是 Attention 聚合后的 768 维向量)
        self.fusion_dropout = nn.Dropout(p=0.5) 
        self.diff_head_k = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, self.DIFF_HIDDEN_DIM), # ✅ 768 -> 384
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.DIFF_HIDDEN_DIM, 1)
        )
        self.norm = nn.LayerNorm(self.FEATURE_DIM * 2)
        # ----------------------------------------------------------------------
        # 个性化 GateNet (输入为 Mean+Max: 1536 维)
        # ----------------------------------------------------------------------
        self.gate_net = nn.Sequential(
            nn.Linear(self.FEATURE_DIM * 2, self.GATE_HIDDEN_DIM), # ✅ 1536 -> 128
            nn.ReLU(),
            nn.Linear(self.GATE_HIDDEN_DIM, 1)
        )
        # --- 关键修改在这里 ---
        
        
        # 最终输出层
        self.output_layer = nn.Linear(knowledge_n, 1)


        print(f"⚡ [Model] 正在将离线特征加载到 GPU 显存: {OUTPUT_FILE} ...")
        cache = torch.load(OUTPUT_FILE, map_location='cpu') # 先读到 CPU
        
        # 假设题目ID是连续的 0 ~ 947。如果不连续，需要你自己做映射表。
        # 我们按 PID 排序，确保索引对齐
        sorted_pids = sorted(list(cache.keys()))
        
        # --- 1. 处理图像特征 (4层) ---
        # 结构: cache[pid]['img'] 是一个 list [L1, L2, L3, L4]
        # 我们要把所有题目的 L1 拼在一起 -> [948, C, H, W]
        print("   正在堆叠图像特征...")
        self.register_buffer('bank_img_l1', torch.stack([cache[p]['img'][0] for p in sorted_pids]))
        self.register_buffer('bank_img_l2', torch.stack([cache[p]['img'][1] for p in sorted_pids]))
        self.register_buffer('bank_img_l3', torch.stack([cache[p]['img'][2] for p in sorted_pids]))
        self.register_buffer('bank_img_l4', torch.stack([cache[p]['img'][3] for p in sorted_pids]))
        
        # --- 2. 处理文本特征 (3层) ---
        print("   正在堆叠文本特征...")
        self.register_buffer('bank_txt_l1', torch.stack([cache[p]['txt'][0] for p in sorted_pids]))
        self.register_buffer('bank_txt_l2', torch.stack([cache[p]['txt'][1] for p in sorted_pids]))
        self.register_buffer('bank_txt_l3', torch.stack([cache[p]['txt'][2] for p in sorted_pids]))
        
        # --- 3. 处理 Mask ---
        # 假设你在提取脚本里加了 mask
        if 'mask' in cache[sorted_pids[0]]:
            print("   正在堆叠 Mask...")
            self.register_buffer('bank_mask', torch.stack([cache[p]['mask'] for p in sorted_pids]))
        else:
            self.bank_mask = None
            
        print("✅ 特征已全部驻留 GPU！")


        self.snr_diff_head = SNRDifficultyHead(feature_dim=256) # 确保维度是 256



        # 假设 feature_dim 是 768
        self.img_proj_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # 降维到 128，算 Loss 更快更准
        )

        self.txt_proj_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.gate = nn.Parameter(torch.tensor(0.0))


        pretrained_matrix = torch.load(KNOW_OUTPUT_FILE, map_location='cpu')
        
        # 3. 创建冻结的 Embedding 层 (作为特征库)
        # freeze=True: 保持 BERT 原味语义，不参与微调 (推荐)
        # freeze=False: 允许 BERT 特征随任务微调 (容易过拟合，不推荐)
        self.know_bert_emb = nn.Embedding.from_pretrained(pretrained_matrix, freeze=True)
        
        # 4. 维度投影层 (768 -> 256)
        # 这个层是随机初始化的，也是之后唯一需要训练的部分
        self.know_projector = nn.Linear(768, 256)

        # 在 __init__ 里
        self.aux_classifier = nn.Linear(256, self.know_bert_emb.num_embeddings) # 86
        # 1. 改良版 Gate: 不变结构，但我们要改它的输出层初始化
        self.se_gate = nn.Sequential(
            nn.Linear(self.knowledge_dim, self.knowledge_dim // 2), # 稍微宽一点，保留更多信息
            nn.LayerNorm(self.knowledge_dim // 2), # 加个 Norm 稳定梯度
            nn.ReLU(),
            nn.Linear(self.knowledge_dim // 2, self.knowledge_dim)
            # 注意：这里去掉了 Sigmoid，我们放在 forward 里加温度控制
        )
        
        # 2. 个性化 Alpha: 允许更大的波动
        self.alpha_net = nn.Linear(self.knowledge_dim, 1)
        self.diff_head_global = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, self.DIFF_HIDDEN_DIM), # ✅ 768 -> 384
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.DIFF_HIDDEN_DIM, 1)
        )
        
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
     
        nn.init.xavier_normal_(self.W_p)
       
        
        # W_p参数 - 增大初始化方差  
        nn.init.normal_(self.W_p, mean=0, std=0.05)  # 从0.02改为0.1
    
       
    
    
        
   

    # ======================================================
    # 🛠️ 使用方法 (在 forward 函数中)
    # ======================================================
    def get_knowledge_embedding(self, knowledge_ids):
        """
        替代原来的 self.know_pro(knowledge_ids)
        input: knowledge_ids [Batch, ...] (例如: 0, 5, 85)
        output: [Batch, ..., 256]
        """
        # 1. 查表获取 768 维 BERT 特征
        bert_feats = self.know_bert_emb(knowledge_ids) # [Batch, 768]
        
        # 2. 投影到 256 维
        final_feats = self.know_projector(bert_feats)  # [Batch, 256]
        
        return final_feats
    def forward(self, batch):
        
        # --- 🔧 定义监控函数 (只在 Rank 0 打印) ---
        def check_mem(tag):
            if torch.distributed.get_rank() == 0 and self.training:
                # 显存单位 GB
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"   💾 [监控] {tag:<20}: {mem:.4f} GB")

        if torch.cuda.is_available():
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            device = torch.device('cpu')
        
        
        stu_id = batch['student_ids'].long().to(device)
        exer_id = batch['problem_ids'].long().to(device)
        kn_emb = batch['knowledges'].to(device)


         # ------------------------------------------------------------------
        # 1. NCDM 基础部分
        # ------------------------------------------------------------------
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        e_discrimination = torch.sigmoid(self.e_discrimination_NCDM(exer_id)) * 10
        k_difficulty = torch.sigmoid(self.k_difficulty_NCDM(exer_id))
        
        pids = batch['problem_ids'].long().to(self.bank_img_l1.device)
        
     
        img_raw_list = [
            self.bank_img_l1[pids], # [Batch, C, H, W]
            self.bank_img_l2[pids],
            self.bank_img_l3[pids],
            self.bank_img_l4[pids]
        ]
        
        txt_raw_list = [
            self.bank_txt_l1[pids],
            self.bank_txt_l2[pids],
            self.bank_txt_l3[pids]
        ]
        
        # 处理 Mask
        if self.bank_mask is not None:
            raw_mask = self.bank_mask[pids]
            padding_mask = (raw_mask == 0)
        else:
            padding_mask = None
        
        # 1. 找出不重复的题目 ID
        # unique_pids: [Unique_Count] (例如 300个)
        # inverse_indices: [Batch_Size] (例如 512个)，记录了原batch每个样本对应第几个unique题目
        unique_pids, inverse_indices = torch.unique(exer_id, sorted=True, return_inverse=True)
        
        # 2. 找出这些唯一题目在原 Batch 中的“代表”位置索引
        # 原理：我们不需要把所有 512 个图都拿来算，只需要拿那 300 个“代表”去算
        perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
        # scatter 操作：后面的索引会覆盖前面的，得到每个 unique ID 最后一次出现的位置
        unique_indices = perm.new_empty(unique_pids.size(0)).scatter_(0, inverse_indices, perm)
        
        # 3. 【筛选】根据索引，只切分出那 300 个样本的特征
        # img_raw_list 是一个列表，里面是 GPU 上的 Tensor，直接切片很快
        unique_img_raw = [t[unique_indices] for t in img_raw_list] 
        unique_txt_raw = [t[unique_indices] for t in txt_raw_list]
        
        # Mask 也要切
        if padding_mask is not None:
            unique_padding_mask = padding_mask[unique_indices]
        else:
            unique_padding_mask = None

        # =========================================================
        # 4. 【计算】特征提取与融合 (只跑 300 次！省算力！省显存！)
        # =========================================================
        
        unique_kn_labels = kn_emb[unique_indices].float() # [Unique_Count, 86]




        # 输入形状: [300, ...] -> 输出形状: [300, 256]
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
                # =======================================================
                # 🚀 核心修改开始
                # =======================================================
                
                # 1. 查表获取 768 维的 BERT 原生特征
                # knowledge_indices 就是 [0, 5, 12...] 这种 ID
                raw_bert_emb = self.know_bert_emb(knowledge_indices) # [K, 768]
                
                # 2. 投影到 256 维
                # 这样 selected_knowledge 就是 [K, 256] 了，和你想要的维度一致
                selected_knowledge = self.know_projector(raw_bert_emb) 
                
                # 归一化 (保持你原有的逻辑)
                selected_knowledge = F.normalize(selected_knowledge, p=2, dim=-1)

                
                
                # 矩阵乘法
                intermediate = torch.matmul(selected_knowledge, W_p_safe)
                intermediate = F.layer_norm(intermediate, intermediate.shape[-1:])
                
               

                selected_F_j = F_j[batch_indices].float()
                
                # Attention 计算
                W_j_selected = torch.bmm(selected_F_j, intermediate.unsqueeze(-1)).squeeze(-1)
                U_j_selected = torch.bmm(W_j_selected.unsqueeze(1), selected_F_j).squeeze(1)
                U_j_selected = F.layer_norm(U_j_selected, U_j_selected.shape[-1:])
                
               
                
                # Dropout + DiffHead
                #U_j_selected = F.dropout(U_j_selected, p=0.3, training=self.training)
                linear_output = self.diff_head_k(U_j_selected)
                selected_difficulty_pred = torch.sigmoid(linear_output)

            # 离开安全区
            selected_difficulty_pred = selected_difficulty_pred.to(modality_k_difficulty.dtype)
            modality_k_difficulty[batch_indices, knowledge_indices] = selected_difficulty_pred.squeeze(1)

        alpha = torch.sigmoid(self.gate) 
        
        
        
       

       
        if self.training:
            mod_diff_drop = F.dropout(modality_k_difficulty, p=0.2) # 30% 概率丢弃特征
        else:
            mod_diff_drop = modality_k_difficulty
        
       
       
      
        # 标准的加权融合，不需要再搞那个 mask_keep_id 了
        f_k_difficulty =1.0* modality_k_difficulty + 0.0* k_difficulty
        
        
        # [A] 交互输入：使用 stu_raw 和 fused_difficulty_logits
        stu_raw = self.student_emb(stu_id)     
        raw_interaction = stu_raw * f_k_difficulty
        
        # [B] 计算门控 Logits
        gate_logits = self.se_gate(f_k_difficulty)
        
        # [C] 温度锐化
        channel_weights = torch.sigmoid(gate_logits * 5.0)
        
        # [D] 加权交互
        clean_interaction = raw_interaction * channel_weights
        
        # [E] 计算个性化敏感度 Alpha
        alpha_logit = self.alpha_net(clean_interaction)
        alpha_sensitivity = 1.0 + 0.4 * torch.tanh(alpha_logit/2.0)
        

        loss_reg = torch.mean(torch.abs(channel_weights - 0.5)) * -0.01 + torch.mean(alpha_logit ** 2) * 0.01
        
        # [F] 注入 NCDM
        # 核心：区分度 * (能力-难度) * 敏感度 * mask
        core_term = stu_emb - f_k_difficulty
        input_x_final = e_discrimination * (core_term * alpha_sensitivity) * kn_emb
        
        #input_x_final = e_discrimination * (stu_emb - f_k_difficulty) * kn_emb
        input_x_final = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_final)))
        input_x_final = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_final)))
        pred_final = self.prednet_full3(input_x_final)

        pred_final = torch.clamp(pred_final, min=-10.0, max=10.0) # 防爆

        #loss_reg=torch.tensor(0.0, device=device)
        return pred_final, loss_reg, loss_reg, loss_reg
        
        '''
        alpha = torch.sigmoid(self.gate) 
        
        
        
       

        # =========================================================
        # 🚀 核心修改：并行计算三条通路 (Multi-Head)
        # =========================================================

        # --- 通路 1: 纯 ID 预测 (保证 ID 能够正常热启动，维持 0.78 的基准) ---
        input_x_id = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x_id = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_id)))
        input_x_id = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_id)))
        pred_id = self.prednet_full3(input_x_id)
        #pred_id = self.output_layer(input_x_id)
        pred_id = torch.clamp(pred_id, min=-10.0, max=10.0) # 防爆

        # --- 通路 2: 纯模态 预测 (强迫图像分支独立干活，不许偷懒！) ---
        # 💡 Trick: 这里加一个 Dropout，防止模态死记硬背 (解决 Visual ID 问题)
        if self.training:
            mod_diff_drop = F.dropout(modality_k_difficulty, p=0.2) # 30% 概率丢弃特征
        else:
            mod_diff_drop = modality_k_difficulty
            
        input_x_img = e_discrimination * (stu_emb - mod_diff_drop) * kn_emb
        #pred_img = self.output_layer(input_x_img)
        input_x_img = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_img)))
        input_x_img = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_img)))
        pred_img = self.prednet_full3(input_x_img)
        pred_img = torch.clamp(pred_img, min=-10.0, max=10.0) # 防爆

        # --- 通路 3: 融合 预测 (最终结果) ---
        alpha = torch.sigmoid(self.gate)
        # 标准的加权融合，不需要再搞那个 mask_keep_id 了
        f_k_difficulty = 0.2 * modality_k_difficulty + 0.8 * k_difficulty
        
        
        # [A] 交互输入：使用 stu_raw 和 fused_difficulty_logits
        stu_raw = self.student_emb(stu_id)     
        raw_interaction = stu_raw * f_k_difficulty
        
        # [B] 计算门控 Logits
        gate_logits = self.se_gate(f_k_difficulty)
        
        # [C] 温度锐化
        channel_weights = torch.sigmoid(gate_logits * 5.0)
        
        # [D] 加权交互
        clean_interaction = raw_interaction * channel_weights
        
        # [E] 计算个性化敏感度 Alpha
        alpha_logit = self.alpha_net(clean_interaction)
        alpha_sensitivity = 1.0 + torch.tanh(alpha_logit) * 2.0 
        
        # =========================================================
        # 🟢 补回 Loss Reg Calculation
        # =========================================================
        # 1. channel_weights - 0.5 的绝对值越大越好 (逼近0或1) -> 乘以负号最小化
        # 2. alpha_logit 越小越好 (防止爆炸)
        loss_reg = torch.mean(torch.abs(channel_weights - 0.5)) * -0.01 + torch.mean(alpha_logit ** 2) * 0.01
        
        # [F] 注入 NCDM
        # 核心：区分度 * (能力-难度) * 敏感度 * mask
        core_term = stu_emb - f_k_difficulty
        input_x_final = e_discrimination * (core_term * alpha_sensitivity) * kn_emb
        
        #input_x_final = e_discrimination * (stu_emb - f_k_difficulty) * kn_emb
        input_x_final = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_final)))
        input_x_final = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_final)))
        pred_final = self.prednet_full3(input_x_final)

        #pred_final = self.output_layer(input_x_final)
        pred_final = torch.clamp(pred_final, min=-10.0, max=10.0) # 防爆

        # =========================================================
        # 🗑️ 清理: 删掉原来那些 mse_loss, loss_img, loss_txt 的计算
        # 我们把 Loss 的计算全部移到外面去，保持 Model 干净
        # =========================================================
        

       
        # 返回 3 个预测值 + Gate值
        # pred_final: 主预测
        # pred_id:    辅助 ID 预测
        # pred_img:   辅助 模态 预测
        return pred_final, pred_id, pred_img, alpha
        '''
       
        '''
        if self.training:
            # 1. 拿到那些 "Unique 题目" 的知识点
            # kn_emb 是全 Batch 的 (比如 512 个)
            # 我们需要去重后的 (比如 300 个)，用来跟 final_img_rep (300个) 对应
            # unique_indices 是你在前面去重步骤里生成的 (就是那个 scatter 之前算出来的索引)
            unique_kn = kn_emb[unique_indices].float() 
            
            # 2. 制作正样本 Mask (只要有 1 个知识点相同，就认为是同类)
            # [300, K] @ [K, 300] -> [300, 300]
            sim_matrix = torch.matmul(unique_kn, unique_kn.T)
            pos_mask = (sim_matrix > 3).float() 
            
            # 3. 算 Loss (调用下面那个函数)
            # final_img_rep 是你在 update_features 里单独吐出来的纯图像特征
            loss_img = self.compute_supcon_loss(final_img_rep, pos_mask)
            
            # final_txt_rep 是纯文本特征
            loss_txt = self.compute_supcon_loss(final_txt_rep, pos_mask)
            
            # 4. 加权得到最终辅助 Loss
            mse_loss = 0.5 * loss_img + 0.5 * loss_txt
        '''
        
        '''
        # 2. 只在训练时计算辅助 Loss
        if self.training:
            # 关键一步：只把【纯图像特征】扔进去预测
            # final_img_rep 是 [Unique_Count, 256]
            k_pred_logits = self.aux_classifier(final_img_rep) 
            
            # 计算 BCE Loss (多标签分类)
            # 让模型学会：这张图里画了什么，就对应什么知识点
            aux_loss = F.binary_cross_entropy_with_logits(k_pred_logits, unique_kn_labels)
            
            # 赋值给 mse_loss
            # (这样你在外面的 train_epoch 里写的 loss = main + 0.2 * mse_loss 就能生效了)
            mse_loss = aux_loss 
        '''
        # 👆👆👆 [修改代码结束] 👆👆👆

        
    def compute_supcon_loss(self, features, mask, temp=0.1):
        """
        计算对比损失
        features: [N, Dim] (例如 [300, 768])
        mask: [N, N] (0/1 矩阵，谁和谁是同类)
        """
        # 1. 归一化 (算余弦相似度必须做)
        features = F.normalize(features, dim=1)
        
        # 2. 算相似度矩阵 [N, N]
        logits = torch.matmul(features, features.T) / temp
        
        # 3. 数值稳定 (减最大值)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 4. 把"自己跟自己"的情况挖掉 (对角线设为0)
        batch_size = features.shape[0]
        # 生成对角线 Mask
        eye_mask = torch.eye(batch_size, device=features.device)
        # 分母 mask: 所有人除了自己
        denominator_mask = 1 - eye_mask
        # 分子 mask: 正样本除了自己
        numerator_mask = mask * denominator_mask
        
        # 5. 算公式
        exp_logits = torch.exp(logits) * denominator_mask
        # log_prob = logits - log(sum(exp))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # 6. 只算正样本的平均 Loss
        # 有正样本的行才算，防止除以 0
        mask_sum = numerator_mask.sum(1)
        mean_log_prob_pos = (numerator_mask * log_prob).sum(1) / (mask_sum + 1e-6)
        
        # 最终 Loss
        loss = -mean_log_prob_pos[mask_sum > 0].mean()
        
        if torch.isnan(loss): return torch.tensor(0.0, device=features.device)
        return loss


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



        
   
    def update_features(self, img_raw_list, txt_raw_list,padding_mask):
        """
        img_raw_list: [B, 256, 56, 56], ... (已在 GPU)
        txt_raw_list: [B, 80, 768], ... (已在 GPU)
        """
      
        
        # 2. 融合 (Fusion)
        fused_out, final_img_rep, final_txt_rep = self.model_feat(img_raw_list, txt_raw_list,padding_mask)

        '''
        # 2. 准备 Checkpoint
        # 只有在训练模式下才开启 Checkpoint
        if self.training:
            
            # ⚠️ 关键动作：因为 img_vecs 是中间变量，
            # 必须显式开启梯度，否则 checkpoint 会报错：
            # "element 0 of tensors does not require grad..."
            for t in img_raw_list:
                t.requires_grad_(True)
            for t in txt_raw_list:
                t.requires_grad_(True)
                
            # 3. 使用 checkpoint 包裹融合层
            # self.fusion (或 self.model_feat) 是你定义的那个大融合模块
            # use_reentrant=False 是 PyTorch 新版推荐写法，更稳定
            fused_out = checkpoint(
                self.model_feat, 
                img_raw_list, 
                txt_raw_list, 
                padding_mask, 
                use_reentrant=False
            )
            
        else:
            # 验证/测试模式，或者不训练时，正常前向传播
            fused_out = self.model_feat(img_raw_list, txt_raw_list, padding_mask)
        '''
        return fused_out, final_img_rep, final_txt_rep
        



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



    def compute_contrastive_loss(self, problem_feat, kn_emb, k_difficulty, exer_id):
        """
        计算自然对应对比损失
        problem_feat: [B, max_len, 768] 模态特征
        kn_emb: [B, knowledge_n] 知识点掩码
        k_difficulty: [B, knowledge_n] 预测的难度
        exer_id: [B] 题目ID
        """
        batch_size = problem_feat.shape[0]
        
        # 1. 提取每个题目的模态表示（通过平均池化）
        modal_embeddings = torch.mean(problem_feat, dim=1)  # [B, 768]
        modal_embeddings = F.normalize(modal_embeddings, p=2, dim=1)  # L2归一化
        
        # 2. 计算模态相似度矩阵
        modal_sim_matrix = torch.mm(modal_embeddings, modal_embeddings.t())  # [B, B]
        
        # 3. 计算难度相似度矩阵 (1 - 难度差异)
        # 使用每个题目主要考察的知识点难度
        primary_knowledge = self.get_exercise_difficulty(kn_emb, k_difficulty)  # [B]
        difficulty_sim_matrix = 1 - torch.abs(
            primary_knowledge.unsqueeze(1) - primary_knowledge.unsqueeze(0)
        )  # [B, B]
        
        # 4. 创建知识点掩码 (同一知识点为1)
        kp_mask = self.create_knowledge_point_mask(kn_emb)  # [B, B]
        
        return self.natural_contrastive_loss(
            modal_sim_matrix, difficulty_sim_matrix, kp_mask
        )

    def get_exercise_difficulty(self, kn_emb, k_difficulty):
        sum_diff = (kn_emb * k_difficulty).sum(dim=1)
        count = kn_emb.sum(dim=1).clamp(min=1)
        return sum_diff / count

    def create_knowledge_point_mask(self, kn_emb):
        kp_mask = (torch.mm(kn_emb, kn_emb.t()) > 0).float()
        kp_mask.fill_diagonal_(0)
        return kp_mask

    def natural_contrastive_loss(self, modal_sim, difficulty_sim, kp_mask):
        """使用这个版本 - 简单快速且有效"""
        modal_sim = (modal_sim + 1) / 2  # 归一化到[0,1]
        valid_mask = kp_mask > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=modal_sim.device)
        return F.mse_loss(modal_sim[valid_mask], difficulty_sim[valid_mask])


    def print_memory(self,tag=""):
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        print(f"[{tag}] allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")





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
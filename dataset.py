import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from configs.dataset_config import *
import torch
import torch_geometric as pyg
from torch_geometric.data import HeteroData
from configs.dataset_config import *
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import h5py
from torch.utils.data import DataLoader
import os
from torch.multiprocessing import get_context  # 新增
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from scipy.stats import pearsonr
import itertools
import torch
import torch.nn as nn  # 添加这行导入
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
from pathlib import Path
import torch.distributed as dist
from configs.dataset_config import * # 假设你的配置都在这里

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch.distributed as dist
from pathlib import Path
from configs.dataset_config import *

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch.distributed as dist
from pathlib import Path
from configs.dataset_config import *

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch.distributed as dist
from pathlib import Path
from configs.dataset_config import *

# ============================================================================
# 1. ProblemDataset (只负责查离线字典)
# ============================================================================
class ProblemDataset(Dataset):
    def __init__(self, feature_cache_path=None):
        """
        :param feature_cache_path: 离线特征文件的路径
        """
        # 使用配置中的路径，或者传入的路径
        self.feature_cache_path = OUTPUT_FILE
        
        self._init_paths()
        self._load_data()
        self._validate_problems()
        self._build_knowledge_dict()

    def _init_paths(self):
        self.knowledge_path = Path(RAW_DATA['knowledge'])

    def _load_data(self):
        # 1. 加载知识点CSV
        print(f"Loading knowledge from {self.knowledge_path}...")
        self.knowledge_df = pd.read_csv(self.knowledge_path)
        '''
        # 2. 加载离线特征 PT 文件
        if not os.path.exists(self.feature_cache_path):
            raise FileNotFoundError(f"找不到离线特征文件: {self.feature_cache_path}，请先运行 run_extraction.py！")

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        print(f"⚡ [Rank {rank}] 正在把离线特征加载进内存: {self.feature_cache_path} ...")
        
        # map_location='cpu' 是关键，不占显存
        self.features_dict = torch.load(self.feature_cache_path, map_location='cpu')
        
        print(f"✅ 特征加载完成，包含 {len(self.features_dict)} 个题目。")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        '''
    def _validate_problems(self):
        knowledge_ids = set(map(int, self.knowledge_df[PROBLEM_ID_COL]))
        
        # 3. 生成有效列表 (直接使用 knowledge_ids)
        self.valid_pids = sorted(
            knowledge_ids,
            key=lambda x: int(x)
        )
        
        print(f"唯一有效题目数量 (仅基于知识点): {len(self.valid_pids)}")

    def _build_knowledge_dict(self):
        print("正在构建知识点字典...")
        self.knowledge_dict = {}
        for pid in self.valid_pids:
            self.knowledge_dict[int(pid)] = self._get_knowledge_vector(pid)

    def _get_knowledge_vector(self, pid):
        try:
            skills = self.knowledge_df[
                self.knowledge_df[PROBLEM_ID_COL] == int(pid)
            ][SKILL_ID_COL].iloc[0]
            
            if isinstance(skills, str):
                skills = eval(str(skills).replace('"', ''))
            
            vector = torch.zeros(TOTAL_SKILLS)
            vector[torch.tensor(skills)] = 1
            return vector
        except Exception as e:
            # print(f"知识点生成失败: {pid}")
            return torch.zeros(TOTAL_SKILLS)

    # --- 供 RecordDataset 调用的接口 ---
    def get_features(self, pid):
        """返回 (img_list, txt_list, mask)"""
        pid = int(pid)
        feats = self.features_dict[pid]
        return feats['img'], feats['txt'], feats['mask']

    def get_knowledge(self, pid):
        return self.knowledge_dict[int(pid)]

    def __len__(self):
        return len(self.valid_pids)

    def __getitem__(self, idx):
        pid = self.valid_pids[idx]
        feats = self.features_dict[pid]
        return {
            'problem_id': int(pid),
            'img_raw': feats['img'], 
            'txt_raw': feats['txt'],
            'txt_mask': feats['mask'],
            'knowledge': self.knowledge_dict[int(pid)]
        }
    
    def get_skill_to_problems(self):
        skill_to_problems = {i: [] for i in range(TOTAL_SKILLS)}
        for pid in self.valid_pids:
            knowledge_vec = self.knowledge_dict[int(pid)]
            for skill_id in torch.nonzero(knowledge_vec).squeeze(1).tolist():
                skill_to_problems[skill_id].append(int(pid))
        return skill_to_problems


# ============================================================================
# 2. RecordDataset (包含自定义 Collate_fn)
# ============================================================================
class RecordDataset(Dataset):
    def __init__(self, mode='train', rank=None):
        self.mode = mode
        # 初始化上面的 ProblemDataset，自动加载内存
        self.problem_data = ProblemDataset(feature_cache_path=OUTPUT_FILE)
        
        print(f"🚀 [RecordDataset] 正在初始化 {mode} 集...")
        self._load_records()
        self._validate_records()
        # self._build_exer_kn_graph() # 按需保留
        self.rank = rank

    def _load_records(self):
        file_path = Path(RAW_DATA[self.mode])
        self.records = pd.read_csv(file_path)
        
        # 兼容不同格式的 USER_ID
        user_ids = set(map(str, self.records[USER_ID_COL]))
        self.user_n = len(user_ids)

    def _validate_records(self):
        valid_pids = set(self.problem_data.valid_pids)
        self.records = self.records[
            self.records[PROBLEM_ID_COL].astype(int).isin(valid_pids)
        ]
        print(f"[{self.mode}] 有效记录: {len(self.records)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]
        pid = int(record[PROBLEM_ID_COL])
        # 确保 student_id 转为 int (如果是 Embedding 需要)
        try:
            raw_stu_id = int(record[USER_ID_COL]) 
        except:
            # 如果 ID 是字符串hash，需要外部处理映射，这里假设是 int
            raw_stu_id = int(record[USER_ID_COL])

        # ✅ 通过 ProblemDataset 获取特征
        #img_raw, txt_raw, txt_mask = self.problem_data.get_features(pid)
        knowledge = self.problem_data.get_knowledge(pid)

        return {
            'student_id': raw_stu_id,
            'problem_id': pid,
            'correct': torch.tensor(record[CORRECT_COL], dtype=torch.float),
            'knowledge': knowledge.clone(),
            
            # ✅ 传递特征列表
            #'img_raw': img_raw, 
            #'txt_raw': txt_raw,
            #'txt_mask': txt_mask
            
        }

    # 🔥🔥🔥 关键：处理特征列表的打包 🔥🔥🔥
    def collate_fn(self, batch):
        student_ids = torch.tensor([x['student_id'] for x in batch])
        problem_ids = torch.tensor([x['problem_id'] for x in batch])
        corrects = torch.stack([x['correct'] for x in batch]).float()
        knowledges = torch.stack([x['knowledge'] for x in batch])

        '''
        # 1. 图像特征: List[List[Tensor]] -> List[BatchTensor]
        # zip(*all_imgs) 会把 layer1 聚合，layer2 聚合...
        all_imgs = [x['img_raw'] for x in batch]
        batch_img_raw = [torch.stack(layers) for layers in zip(*all_imgs)]
        
        # 2. 文本特征: List[List[Tensor]] -> List[BatchTensor]
        all_txts = [x['txt_raw'] for x in batch]
        batch_txt_raw = [torch.stack(layers) for layers in zip(*all_txts)]

        # 3. Mask: List[Tensor] -> BatchTensor
        batch_txt_mask = torch.stack([x['txt_mask'] for x in batch])
        '''
        return {
            'student_ids': student_ids,
            'problem_ids': problem_ids,
            'corrects': corrects,
            'knowledges': knowledges,
            
            #'img_raw': batch_img_raw, 
            #'txt_raw': batch_txt_raw,
            #'txt_mask': batch_txt_mask # [Batch, 80]
            
        }

    def create_dataloader(self, sampler, batch_size, num_workers):
        # 自动调整参数
        prefetch_factor = 2 if num_workers > 0 else None
        persistent_workers = True if num_workers > 0 else False
        
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True, # GPU驻留的话其实这个也可以False，不过True也没事
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers, # ✅ 必须和 num_workers>0 配合
            prefetch_factor=prefetch_factor # ✅ 必须和 num_workers>0 配合
        )
from collections import defaultdict
import random
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedBalancedProblemBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, max_problems=200, num_replicas=None, rank=None, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_problems = max_problems  # 每个 batch 最多 200 个题目

        # 分布式设置
        if num_replicas is None:
            if not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        self.num_replicas = num_replicas

        if rank is None:
            if not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
        self.rank = rank

        self.seed = seed
        self.epoch = 0

        # 构建题目到索引的映射（假设 dataset 有 PROBLEM_ID_COL 字段）
        self.problem_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            pid = dataset.records.iloc[idx][PROBLEM_ID_COL]  # 替换为你的题目 ID 列名
            self.problem_to_indices[pid].append(idx)
        self.unique_problems = list(self.problem_to_indices.keys())

        # 计算每个 rank 的 batch 数量
        self.num_samples = len(dataset) // self.num_replicas  # 每 GPU 的样本数
        self.total_size = self.num_samples * self.num_replicas  # 全局样本数（对齐）

    def set_epoch(self, epoch):
        self.epoch = epoch  # 用于同步不同 epoch 的随机状态

    def __iter__(self):
        # 设置随机种子（确保分布式环境下各进程同步）
        g = random.Random()
        g.seed(self.seed + self.epoch)

        # 计算每个 rank 的 batch 数量
        total_batches = (len(self.dataset) // self.batch_size) // self.num_replicas

        for _ in range(total_batches):
            # 1. 随机选择最多 200 个题目
            selected_pids = g.sample(
                self.unique_problems,
                min(self.max_problems, len(self.unique_problems))
            )

            # 2. 计算每个题目应该贡献多少样本（尽量均匀分布）
            samples_per_problem = max(1, self.batch_size // len(selected_pids))

            # 3. 从每个题目中抽取 samples_per_problem 条记录
            batch_indices = []
            for pid in selected_pids:
                indices = self.problem_to_indices[pid]
                batch_indices.extend(
                    g.choices(indices, k=samples_per_problem)
                )

            # 4. 如果样本数不足 batch_size，从已选题目中随机补齐
            if len(batch_indices) < self.batch_size:
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(
                    g.choices(batch_indices, k=remaining)  # 从本 batch 已选样本中随机重复
                )

            # 5. 确保 batch_size 对齐 num_replicas（分布式训练）
            while len(batch_indices) % self.num_replicas != 0:
                batch_indices.append(g.choice(batch_indices))  # 随机补一个

            # 6. 按 rank 分配数据（分布式）
            indices_per_rank = len(batch_indices) // self.num_replicas
            start = self.rank * indices_per_rank
            end = start + indices_per_rank

            yield batch_indices[start:end]

    def __len__(self):
        return (len(self.dataset) // self.batch_size) // self.num_replicas


# 新增全局数据集加载
class FullDataset:
    def __init__(self):
        self.problem_data = ProblemDataset(OUTPUT_FILE)  # 题目数据（包含所有题目）
        self.train_records = RecordDataset(mode='train')
        self.val_records = RecordDataset(mode='val')
        self.test_records = RecordDataset(mode='test')
        
    def get_all_records(self):
        """整合所有答题记录"""
        return pd.concat([
            self.train_records.records,
            self.val_records.records,
            self.test_records.records
        ])


class RelationBuilder:
    def __init__(self, problem_dataset, full_dataset):
        self.problem_data = problem_dataset
        self.record_data = full_dataset.get_all_records()  # 关键修改
        self.hetero_graph = HeteroData()


    def build_graph(self):
        self.hetero_graph.edge_types = [
            ('problem', 'has_knowledge', 'knowledge'),
            ('problem', 'related', 'problem'),
            ('knowledge', 'correlate', 'knowledge')
          
        ]
        self._build_problem_concept_edges()
        # 添加节点特征和构建边（原有逻辑不变）
        #self._add_problem_nodes(FUSION_FEATURES_PATH)
        
        return self.hetero_graph



    '''
    def _add_problem_nodes(self, feature_dir):
        """
        最小化检查版本：仅加载特征并计算知识点均值，输出关键维度信息
        """
        # 1. 确保knowledge节点有明确的num_nodes
        num_knowledge = TOTAL_SKILLS
        self.hetero_graph['knowledge'].num_nodes = num_knowledge

        # 2. 加载题目特征
        pids = sorted(int(pid) for pid in self.problem_data.valid_pids)
        feats = [torch.load(os.path.join(feature_dir, f"{pid}.pt")).detach().numpy() for pid in pids]

        problem_feats = torch.from_numpy(np.array(feats)).float()
        self.hetero_graph['problem'].x = problem_feats
        print(f"已加载 {len(pids)} 个题目特征 | 维度: {problem_feats.shape}")

        # 3. 计算知识点平均特征
        edge_index = self.hetero_graph['problem', 'has_knowledge', 'knowledge'].edge_index
        
        # 计算均值
        knowledge_feats = torch.zeros(
            (num_knowledge, problem_feats.size(1)),
            device=problem_feats.device
        )
        knowledge_feats.scatter_add_(
            0,
            edge_index[1].unsqueeze(-1).expand(-1, problem_feats.size(1)),
            problem_feats[edge_index[0]]
        )
        
        # 修正点：统一数据类型
        degree = torch.zeros(num_knowledge, 
                        device=edge_index.device,
                        dtype=torch.float32)  # 明确使用float32
        degree.scatter_add_(0, 
                        edge_index[1], 
                        torch.ones_like(edge_index[1], dtype=torch.float32))  # 确保类型匹配
        
        self.hetero_graph['knowledge'].x = knowledge_feats / degree.unsqueeze(-1)
        print(f"已计算 {num_knowledge} 个知识点特征 | 维度: {knowledge_feats.shape}")
    '''
    def _add_problem_nodes(self, feature_dir):
        """
        最终修改版本：避免使用register_buffer
        """
        
        feature_dir = TEXT_FEATURES_DIR
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 确保knowledge节点定义
        num_knowledge = TOTAL_SKILLS
        self.hetero_graph['knowledge'].num_nodes = num_knowledge

        # 初始化随机投影矩阵（设备感知）
        if not hasattr(self, 'random_projection'):
            input_dim = 256 * 768
            output_dim = 512
            # 直接保存张量（无需register_buffer）
            self.random_projection = torch.randn(input_dim, output_dim, device=device) * (2.0 / input_dim)**0.5
            
        # 加载题目特征（保持原逻辑）
        pids = sorted(int(pid) for pid in self.problem_data.valid_pids)
        
        # 假设每个 `feat` 原始是 [256, 768]
        with torch.no_grad():
            all_feats = []
            for pid in pids:
                feat = torch.load(os.path.join(feature_dir, f"{pid}.pt")).to(device)  # 原始 [256, 768]


                # 线性变换：直接将 [256, 768] 转换为 [256, 1024]
                
                linear_transform = torch.nn.Linear(768, 512).to(device)  # 注意这里的输入是 768，输出是 1024
                feat = linear_transform(feat)  # 经过线性变换，变成 [256, 1024]
                
                # 平均池化到最终的 [1024]
                feat = feat.mean(dim=0)  # [1024]

                all_feats.append(feat)

            problem_feats = torch.stack(all_feats).to(device)  # [num_problems, 1024]



        # 转换维度（显式确保设备一致）
        #with torch.no_grad():
        #    orig_feats = torch.stack(all_feats).to(device)  # 显式指定设备
        #    flattened = orig_feats.view(-1, 256*768)
            # 确保投影矩阵在相同设备
       #     problem_feats = torch.matmul(flattened, self.random_projection.to(device))
       
        
        self.hetero_graph['problem'].x = problem_feats
        print(f"已加载 {len(pids)} 个题目特征 | 转换后维度: {problem_feats.shape}")
        


        num_knowledge = TOTAL_SKILLS
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载知识点特征（直接按ID顺序）
        knowledge_features = torch.load(os.path.join(KNOW_PT))  # 你的 OUTPUT_PATH
        feature_matrix = torch.stack([knowledge_features[i] for i in range(num_knowledge)]).to(device)

        # 替换原有的 scatter_add 计算（如果不需要聚合题目特征）
        self.hetero_graph['knowledge'].x = feature_matrix
        print(f"直接加载 {num_knowledge} 个知识点特征 | 维度: {feature_matrix.shape}")
        
        
    '''
    def _add_problem_nodes(self, feature_dir):
        """
        最终修改版本：避免使用register_buffer
        """
        feature_dir = TEXT_FEATURES_DIR
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 确保knowledge节点定义
        num_knowledge = TOTAL_SKILLS
        self.hetero_graph['knowledge'].num_nodes = num_knowledge

        # 初始化随机投影矩阵（设备感知）
        if not hasattr(self, 'random_projection'):
            input_dim = 256 * 768
            output_dim = 512
            # 直接保存张量（无需register_buffer）
            self.random_projection = torch.randn(input_dim, output_dim, device=device) * (2.0 / input_dim)**0.5
            
        # 加载题目特征（保持原逻辑）
        pids = sorted(int(pid) for pid in self.problem_data.valid_pids)
        
        # 假设每个 `feat` 原始是 [256, 768]
        with torch.no_grad():
            all_feats = []
            for pid in pids:
                feat = torch.load(os.path.join(feature_dir, f"{pid}.pt")).to(device)  # 原始 [256, 768]


                # 线性变换：直接将 [256, 768] 转换为 [256, 1024]
                linear_transform = torch.nn.Linear(768, 512).to(device)  # 注意这里的输入是 768，输出是 1024
                feat = linear_transform(feat)  # 经过线性变换，变成 [256, 1024]

                # 平均池化到最终的 [1024]
                feat = feat.mean(dim=0)  # [1024]

                all_feats.append(feat)

            problem_feats = torch.stack(all_feats).to(device)  # [num_problems, 1024]



        # 转换维度（显式确保设备一致）
        #with torch.no_grad():
        #    orig_feats = torch.stack(all_feats).to(device)  # 显式指定设备
        #    flattened = orig_feats.view(-1, 256*768)
            # 确保投影矩阵在相同设备
       #     problem_feats = torch.matmul(flattened, self.random_projection.to(device))
       
        
        self.hetero_graph['problem'].x = problem_feats
        print(f"已加载 {len(pids)} 个题目特征 | 转换后维度: {problem_feats.shape}")

        # 知识点特征计算（保持原逻辑不变）
        edge_index = self.hetero_graph['problem', 'has_knowledge', 'knowledge'].edge_index.to(device)
        
        knowledge_feats = torch.zeros(
            (num_knowledge, 512),
            device=device
        )
        knowledge_feats.scatter_add_(
            0,
            edge_index[1].unsqueeze(-1).expand(-1, 512),
            problem_feats[edge_index[0]]
        )
        
        degree = torch.zeros(num_knowledge, device=device)
        degree.scatter_add_(
            0,
            edge_index[1],
            torch.ones(edge_index.size(1), device=device)
        )
        degree = degree.clamp(min=1)
        
        self.hetero_graph['knowledge'].x = (knowledge_feats / degree.unsqueeze(-1)).to(device)
        print(f"已计算 {num_knowledge} 个知识点特征 | 维度: {knowledge_feats.shape}")
    '''
    '''
    def _add_problem_nodes(self, feature_dir):
        """
        最终修改版本：仅加载题目原始特征 [256, 768]，不做映射
        """
        feature_dir = TEXT_FEATURES_DIR
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_knowledge = TOTAL_SKILLS
        self.hetero_graph['knowledge'].num_nodes = num_knowledge

        # 加载原始题目特征 [256, 768]
        pids = sorted(int(pid) for pid in self.problem_data.valid_pids)
        all_feats = []

        with torch.no_grad():
            for pid in pids:
                feat = torch.load(os.path.join(feature_dir, f"{pid}.pt")).to(device)  # [256, 768]
                all_feats.append(feat)

        # 直接保存原始特征（不flatten、不变换）
        problem_feats = torch.stack(all_feats)  # [num_problems, 256, 768]
        self.hetero_graph['problem'].x = problem_feats
        print(f"已加载 {len(pids)} 个题目原始特征 | 维度: {problem_feats.shape}")

        # 计算知识点特征（保持原逻辑）
        edge_index = self.hetero_graph['problem', 'has_knowledge', 'knowledge'].edge_index.to(device)

        # 平均池化题目特征用于知识点
        pooled_feats = problem_feats.mean(dim=1)  # [num_problems, 768]

        knowledge_feats = torch.zeros((num_knowledge, 768), device=device)
        knowledge_feats.scatter_add_(
            0,
            edge_index[1].unsqueeze(-1).expand(-1, 768),
            pooled_feats[edge_index[0]]
        )

        degree = torch.zeros(num_knowledge, device=device)
        degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))
        degree = degree.clamp(min=1)

        self.hetero_graph['knowledge'].x = (knowledge_feats / degree.unsqueeze(-1)).to(device)
        print(f"已计算 {num_knowledge} 个知识点特征 | 维度: {knowledge_feats.shape}")

    '''
    '''
    def _add_problem_nodes(self, feature_dir):
        """
        最终修改版本：避免使用register_buffer
        """
        feature_dir = IMAGE_FEATURES_DIR
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 确保knowledge节点定义
        num_knowledge = TOTAL_SKILLS
        self.hetero_graph['knowledge'].num_nodes = num_knowledge

      
            
        # 加载题目特征（保持原逻辑）
        pids = sorted(int(pid) for pid in self.problem_data.valid_pids)
        
        # 假设每个 feat 原始是 [512, 56, 56]
        with torch.no_grad():
            all_feats = []
            for pid in pids:
                feat = torch.load(os.path.join(feature_dir, f"{pid}.pt")).to(device)  # 原始 [512, 56, 56]

                # 假设是卷积输入，我们可以通过一个卷积层将 [512, 56, 56] 转换为 [1024]
                # 这里先用卷积层来处理
                conv_layer = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1).to(device)  # [512, 56, 56] -> [1024, 56, 56]
                feat = conv_layer(feat)  # 经过卷积，变成 [1024, 56, 56]

                # 进行池化操作，将 [1024, 56, 56] 池化为 [1024, 1, 1]
                feat = nn.AdaptiveAvgPool2d(1)(feat)  # 使用自适应池化，输出 [1024, 1, 1]

                # 展平（flatten）为 [1024]
                feat = feat.view(-1)  # 将 [1024, 1, 1] 展平为 [1024]

                all_feats.append(feat)

            problem_feats = torch.stack(all_feats).to(device)  # [num_problems, 1024]



        # 转换维度（显式确保设备一致）
        #with torch.no_grad():
        #    orig_feats = torch.stack(all_feats).to(device)  # 显式指定设备
        #    flattened = orig_feats.view(-1, 256*768)
            # 确保投影矩阵在相同设备
       #     problem_feats = torch.matmul(flattened, self.random_projection.to(device))
       
        
        self.hetero_graph['problem'].x = problem_feats
        print(f"已加载 {len(pids)} 个题目特征 | 转换后维度: {problem_feats.shape}")

        # 知识点特征计算（保持原逻辑不变）
        edge_index = self.hetero_graph['problem', 'has_knowledge', 'knowledge'].edge_index.to(device)
        
        knowledge_feats = torch.zeros(
            (num_knowledge, 1024),
            device=device
        )
        knowledge_feats.scatter_add_(
            0,
            edge_index[1].unsqueeze(-1).expand(-1, 1024),
            problem_feats[edge_index[0]]
        )
        
        degree = torch.zeros(num_knowledge, device=device)
        degree.scatter_add_(
            0,
            edge_index[1],
            torch.ones(edge_index.size(1), device=device)
        )
        degree = degree.clamp(min=1)
        
        self.hetero_graph['knowledge'].x = (knowledge_feats / degree.unsqueeze(-1)).to(device)
        print(f"已计算 {num_knowledge} 个知识点特征 | 维度: {knowledge_feats.shape}")
    '''
    def _build_problem_concept_edges(self):
        """构建题目-知识点边（修正版本）"""
        edge_index = []
        for pid in self.problem_data.valid_pids:
            k_vector = self.problem_data.knowledge_dict[int(pid)]
            
            # 修正索引提取逻辑
            indices = torch.where(k_vector == 1)[0]  # 提取满足条件的索引张量
            knowledge_ids = indices.tolist()         # 转换为列表
            
            for k_id in knowledge_ids:
                edge_index.append([int(pid), k_id])
        
        
        edge_index_tensor = torch.tensor(edge_index).t().contiguous()
        self.hetero_graph['problem', 'has_knowledge', 'knowledge'].edge_index = edge_index_tensor
        self.hetero_graph['problem', 'has_knowledge', 'knowledge'].edge_attr = torch.ones(edge_index_tensor.size(1))    
        edge_index = self.hetero_graph['problem', 'has_knowledge', 'knowledge'].edge_index
        print("目标节点索引最大值:", edge_index[1].max().item())  # 应输出82
   
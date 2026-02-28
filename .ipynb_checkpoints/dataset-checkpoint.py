import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import torch
import torch_geometric as pyg
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import os
from torch.multiprocessing import get_context  
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from scipy.stats import pearsonr
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.distributed as dist
from pathlib import Path
from configs.dataset_config import *


class ProblemDataset(Dataset):
    def __init__(self, feature_cache_path=None):
       
        self.feature_cache_path = OUTPUT_FILE
        
        self._init_paths()
        self._load_data()
        self._validate_problems()
        self._build_knowledge_dict()

    def _init_paths(self):
        self.knowledge_path = Path(RAW_DATA['knowledge'])

    def _load_data(self):
        self.knowledge_df = pd.read_csv(self.knowledge_path)
      
    def _validate_problems(self):
        knowledge_ids = set(map(int, self.knowledge_df[PROBLEM_ID_COL]))
        
        self.valid_pids = sorted(
            knowledge_ids,
            key=lambda x: int(x)
        )
       

    def _build_knowledge_dict(self):
       
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
           
            return torch.zeros(TOTAL_SKILLS)

  
    def get_features(self, pid):
       
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



class RecordDataset(Dataset):
    def __init__(self, mode='train', rank=None):
        self.mode = mode
      
        self.problem_data = ProblemDataset(feature_cache_path=OUTPUT_FILE)
        
        self._load_records()
        self._validate_records()
        self.rank = rank

    def _load_records(self):
        file_path = Path(RAW_DATA[self.mode])
        self.records = pd.read_csv(file_path)
        
        user_ids = set(map(str, self.records[USER_ID_COL]))
        self.user_n = len(user_ids)

    def _validate_records(self):
        valid_pids = set(self.problem_data.valid_pids)
        self.records = self.records[
            self.records[PROBLEM_ID_COL].astype(int).isin(valid_pids)
        ]
       

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]
        pid = int(record[PROBLEM_ID_COL])
       
        try:
            raw_stu_id = int(record[USER_ID_COL]) 
        except:
           
            raw_stu_id = int(record[USER_ID_COL])

       
        knowledge = self.problem_data.get_knowledge(pid)

        return {
            'student_id': raw_stu_id,
            'problem_id': pid,
            'correct': torch.tensor(record[CORRECT_COL], dtype=torch.float),
            'knowledge': knowledge.clone(),
            
         
        }

    def collate_fn(self, batch):
        student_ids = torch.tensor([x['student_id'] for x in batch])
        problem_ids = torch.tensor([x['problem_id'] for x in batch])
        corrects = torch.stack([x['correct'] for x in batch]).float()
        knowledges = torch.stack([x['knowledge'] for x in batch])

       
        return {
            'student_ids': student_ids,
            'problem_ids': problem_ids,
            'corrects': corrects,
            'knowledges': knowledges,
            
         
        }

    def create_dataloader(self, sampler, batch_size, num_workers):
        prefetch_factor = 2 if num_workers > 0 else None
        persistent_workers = True if num_workers > 0 else False
        
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers, 
            prefetch_factor=prefetch_factor
        )
from collections import defaultdict
import random
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedBalancedProblemBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, max_problems=200, num_replicas=None, rank=None, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_problems = max_problems 

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

        
        self.problem_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            pid = dataset.records.iloc[idx][PROBLEM_ID_COL] 
            self.problem_to_indices[pid].append(idx)
        self.unique_problems = list(self.problem_to_indices.keys())

      
        self.num_samples = len(dataset) // self.num_replicas  
        self.total_size = self.num_samples * self.num_replicas  

    def set_epoch(self, epoch):
        self.epoch = epoch  

    def __iter__(self):
       
        g = random.Random()
        g.seed(self.seed + self.epoch)

        total_batches = (len(self.dataset) // self.batch_size) // self.num_replicas

        for _ in range(total_batches):
            selected_pids = g.sample(
                self.unique_problems,
                min(self.max_problems, len(self.unique_problems))
            )

            samples_per_problem = max(1, self.batch_size // len(selected_pids))

            batch_indices = []
            for pid in selected_pids:
                indices = self.problem_to_indices[pid]
                batch_indices.extend(
                    g.choices(indices, k=samples_per_problem)
                )

            if len(batch_indices) < self.batch_size:
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(
                    g.choices(batch_indices, k=remaining)  # 从本 batch 已选样本中随机重复
                )

            while len(batch_indices) % self.num_replicas != 0:
                batch_indices.append(g.choice(batch_indices))  # 随机补一个

            indices_per_rank = len(batch_indices) // self.num_replicas
            start = self.rank * indices_per_rank
            end = start + indices_per_rank

            yield batch_indices[start:end]

    def __len__(self):
        return (len(self.dataset) // self.batch_size) // self.num_replicas



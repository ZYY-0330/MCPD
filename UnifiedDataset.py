import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import json
import pandas as pd
from PIL import Image
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Dict, Union
from configs.dataset_config import *
from torch.nn import functional as F
from ast import literal_eval

class UnifiedDataset(Dataset):
    """统一数据集：整合正样本和负样本的原始数据"""
    def __init__(self, 
                 text_path: Path, 
                 knowledge_path: Path, 
                 image_dir: Path, 
                 num_concepts: int, 
                 transform=None):
        # 初始化参数
        self.transform = transform
        self.num_concepts = num_concepts

        
        """初始化文件路径"""
        self.image_dir = image_dir
        self.text_path = text_path
        self.knowledge_path = knowledge_path

        self._load_raw_data()
        self._validate_problems()
        self._build_knowledge_dict()
 
    def _load_raw_data(self):
        """加载原始数据（强制使用整数键版本）"""
        # 加载文本数据并转换键类型
        with open(self.text_path) as f:
            raw_text = json.load(f)
            self.text_data = {
                int(k): v for k, v in raw_text.items()  # 关键转换
            }
        
        # 知识点数据保持整数类型（假设 knowledge_df 的 problem_id 列是整数）
        self.knowledge_df = pd.read_csv(self.knowledge_path)
        
        # 图片路径映射也使用整数键
        self.image_files = {
            int(f.stem): f for f in self.image_dir.glob("*.png")  # 关键转换
        }

    def _validate_problems(self):
        """验证数据一致性（修复版本）"""
        # 获取各数据源的 PID 集合
        text_ids = set(self.text_data.keys())  # 整数集合
        image_ids = set(self.image_files.keys())  # 整数集合
        knowledge_ids = set(map(int, self.knowledge_df[PROBLEM_ID_COL]))  # 修复：转换为整数集合
        
        # 计算交集并排序
        self.valid_pids = sorted(text_ids & image_ids & knowledge_ids)
        
        print(f"唯一有效题目数量: {len(self.valid_pids)}")

    def _build_knowledge_dict(self):
        """预生成知识点向量字典"""
        self.knowledge_dict = {}
        for pid in self.valid_pids:
            self.knowledge_dict[int(pid)] = self._get_knowledge_vector(pid)

    def _get_knowledge_vector(self, pid):
        """生成知识点向量（最终修复版）"""
        try:
            # 1. 精确提取单个值
            skill_value = self.knowledge_df.loc[
                self.knowledge_df[PROBLEM_ID_COL] == int(pid),
                SKILL_ID_COL
            ].iloc[0]  # 关键修复点：使用 .iloc[0] 获取标量值

            # 2. 统一数据类型处理
            if isinstance(skill_value, str):
                # 处理 "[1,2,3]" 或 "1,3" 等格式
                cleaned = skill_value.strip("[]").replace("'", "").replace('"', "")
                skills = list(map(int, cleaned.split(','))) if cleaned else []
            elif isinstance(skill_value, (int, float)):
                skills = [int(skill_value)]
            else:
                raise TypeError(f"未知技能数据类型: {type(skill_value)}")

            # 3. 生成向量
            vector = torch.zeros(self.num_concepts, dtype=torch.float)
            if skills:  # 非空检查
                vector[torch.tensor(skills)] = 1.0
            return vector

        except IndexError:
            # PID 不存在时的处理
            print(f"警告: PID {pid} 不存在于知识库中")
            return torch.zeros(self.num_concepts)
        except Exception as e:
            print(f"知识点生成失败: {pid} - {str(e)}")
            return torch.zeros(self.num_concepts)

    def __len__(self):
        """返回数据集大小"""
        return len(self.valid_pids)

    def __getitem__(self, index):
        """根据索引返回数据样本"""
        # 获取对应的 pid
        pid = self.valid_pids[index]
        
        # 获取文本数据 - 从字典中提取content字段
        text_item = self.text_data[pid]
        
        # 处理字典类型的文本数据
        if isinstance(text_item, dict) and 'content' in text_item:
            text = text_item['content']
        elif isinstance(text_item, str):
            text = text_item
        
        
        # 清理文本
        text = text.strip()
        
        # 获取图片数据
        image_path = self.image_files[pid]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 获取知识点向量
        knowledge_vector = self.knowledge_dict[pid]
        
        return {
            'pid': pid,
            'image': image,
            'text': text,  # 现在是从content字段提取的纯文本
            'knowledge': knowledge_vector
        }
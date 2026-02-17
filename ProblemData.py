from torch.utils.data import Dataset
import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from goal_Eedi import *  # 确保包含 PROBLEM_ID_COL, SKILL_ID_COL 等常量
import ast
from collections import defaultdict
import random

class ProblemDataLoader(Dataset):
    def __init__(self, json_file, image_dir, transform):
        """
        Args:
            json_file (str): JSON文件路径，包含题目文本数据
            image_dir (str): 图片存储目录
            transform (callable): 图像增强变换
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # 加载文本数据
        with open(json_file, 'r') as f:
            text_data = json.load(f)
        self.text_data = {k: v['content'] for k, v in text_data.items()}
        self.problem_ids = list(self.text_data.keys())
        
        # 加载知识点映射数据
        self.knowledge_df = pd.read_csv(Q_FILE)
        self.question_to_subjects = self._parse_knowledge_data()

    def _parse_knowledge_data(self):
        """解析知识点映射关系"""
        
        # 创建一个空字典，存储题目ID到知识点的映射
        knowledge_dict = {}
        
        # 遍历 DataFrame 中的每一行，处理每个问题的知识点数据
        for _, row in self.knowledge_df.iterrows():
            # 提取题目的ID，确保转换为整数类型
            problem_id = int(row[PROBLEM_ID_COL])
            
            try:
                # 尝试解析技能ID字段为字面量 Python 表达式（例如，字符串 '[1, 2]' 会被解析为列表 [1, 2]）
                subjects = ast.literal_eval(row[SKILL_ID_COL])
                
                # 如果解析后是一个整数（即只有一个技能ID），将其转换为包含该技能ID的元组
                if isinstance(subjects, int):
                    subjects = (subjects,)
            except:
                # 如果解析失败（可能是数据格式问题），则直接按逗号分割并转为整数列表
                subjects = tuple(map(int, str(row[SKILL_ID_COL]).split(',')))
            
            # 将处理后的技能ID列表（按升序排序）保存到字典中，键为题目ID，值为对应的技能ID元组
            knowledge_dict[problem_id] = tuple(sorted(subjects))
        
        # 返回包含题目ID和对应知识点的字典
        return knowledge_dict

    def __len__(self):
        return len(self.problem_ids)

    def __getitem__(self, idx):
        problem_id = self.problem_ids[idx]
        
        # 加载图像
        img_path = os.path.join(self.image_dir, f"{problem_id}.png")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 加载文本
        text = self.text_data[problem_id]
        
        # 获取知识点标签（关键修改点）
        int_pid = int(problem_id)
        #subjects = self.question_to_subjects.get(int_pid, (-1,))  # -1表示缺失值
        
        return {
            'problem_id': problem_id,
            'image': image,
            'text': text,
            #'subjects': torch.tensor(subjects, dtype=torch.long)  # 添加subjects字段
        }



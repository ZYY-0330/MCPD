
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from configs.dataset_config import *
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from configs.dataset_config import *
import os
from configs.dataset_config import *

from transformers import BertTokenizer, BertModel
import torch
from torch.amp import autocast
import torch.nn as nn

class MathBERTTextFeatureExtractor(nn.Module):
    def __init__(self, device='auto', seq_length=80):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == 'auto' else device
        )
        self.seq_length = seq_length

        # ✅ 加载模型和 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            MODEL_PATH,
            mathVocab=True
        )
        self.bert_model = BertModel.from_pretrained(
            MODEL_PATH,
            output_hidden_states=True
        ).to(self.device)

        # ✅ 层数 & 选择层
        self.num_layers = len(self.bert_model.encoder.layer)
        self.selected_layers = [0, 6,  self.num_layers]

        # ✅ projection for each layer
        self.projections = nn.ModuleList([
            nn.Linear(768, 768) for _ in self.selected_layers
        ])

        self.projections = self.projections.to(self.device)

        # ✅ 配置训练
        self._configure_for_training()
        self._init_weights()  # 放在 __init__() 的最后
    def _init_weights(self):
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)  # Linear层通常用Xavier
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0)

    def _configure_for_training(self):
        """配置训练层与 checkpoint"""
        '''
        # 只训练目标层，其他冻结
        for name, param in self.bert_model.named_parameters():
            if not any(f'layer.{i}.' in name for i in self.selected_layers):
                param.requires_grad = False

        # ✅ 启用 gradient checkpointing（节省显存）
        if hasattr(self.bert_model, 'gradient_checkpointing_enable'):
            self.bert_model.gradient_checkpointing_enable()

        # ⚠️ 不设置 eval，这样主流程外部可以自己调用 train() 或 eval()
        # 如果你想默认 train 模式：
        self.bert_model.train()
        '''
        for param in self.bert_model.parameters():
            param.requires_grad = False


    def _preprocess_batch(self, texts):
        """将文本转为 token ids"""
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.seq_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def process_batch(self, batch_data, use_amp=True):
        pids, texts = zip(*batch_data)
        inputs = self._preprocess_batch(texts)
        
        # 保存mask供后续使用
        attention_mask = inputs['attention_mask']
        
        if use_amp:
            with autocast(device_type='cuda',enabled=torch.cuda.is_available()):
                outputs = self.bert_model(**inputs, output_hidden_states=True)
        else:
            outputs = self.bert_model(**inputs, output_hidden_states=True)

        projected_feats = []
        for idx, layer_id in enumerate(self.selected_layers):
            if layer_id < len(outputs.hidden_states):
                feat = outputs.hidden_states[layer_id]
                if not feat.requires_grad:
                    feat.requires_grad_()
                feat.retain_grad()
                projected_feats.append(feat)

        # 返回特征和mask
        return projected_feats, attention_mask




    def save_features(self, batch_data):
        features = self.process_batch(batch_data)  # [layer0, layer1, layer2]
        pids, _ = zip(*batch_data)
        
        for i, pid in enumerate(pids):
            for layer_idx, layer_feature in enumerate(features):
                # 保存为.npy
                arr = layer_feature[i].cpu().numpy()
                npy_path = os.path.join(TEXT_FEATURES_DIR, f"{pid}_{layer_idx}.npy")
                np.save(npy_path, arr.astype(np.float16))  # 半精度
                
                # 保留原.pt格式
                pt_path = os.path.join(TEXT_FEATURES_DIR, f"{pid}_{layer_idx}.pt")
                torch.save(layer_feature[i].cpu(), pt_path)

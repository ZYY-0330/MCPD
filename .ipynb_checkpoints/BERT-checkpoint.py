
import torch
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

        self.tokenizer = BertTokenizer.from_pretrained(
            MODEL_PATH,
            mathVocab=True
        )
        self.bert_model = BertModel.from_pretrained(
            MODEL_PATH,
            output_hidden_states=True
        ).to(self.device)

        self.num_layers = len(self.bert_model.encoder.layer)
        self.selected_layers = [0, 6,  self.num_layers]

        self.projections = nn.ModuleList([
            nn.Linear(768, 768) for _ in self.selected_layers
        ])

        self.projections = self.projections.to(self.device)

        self._configure_for_training()
        self._init_weights()  
    def _init_weights(self):
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight) 
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0)

    def _configure_for_training(self):
     
        for param in self.bert_model.parameters():
            param.requires_grad = False


    def _preprocess_batch(self, texts):
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

        return projected_feats, attention_mask




    def save_features(self, batch_data):
        features = self.process_batch(batch_data)  # [layer0, layer1, layer2]
        pids, _ = zip(*batch_data)
        
        for i, pid in enumerate(pids):
            for layer_idx, layer_feature in enumerate(features):
                arr = layer_feature[i].cpu().numpy()
                npy_path = os.path.join(TEXT_FEATURES_DIR, f"{pid}_{layer_idx}.npy")
                np.save(npy_path, arr.astype(np.float16)) 
                
                pt_path = os.path.join(TEXT_FEATURES_DIR, f"{pid}_{layer_idx}.pt")
                torch.save(layer_feature[i].cpu(), pt_path)

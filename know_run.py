import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import json
import os
from tqdm import tqdm

from configs.dataset_config import *

class KnowledgeExtractor:
    def __init__(self, model_path, device):
        self.device = device
        print(f"🚀 Loading BERT from {model_path}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to(device)
        self.model.eval()

    def extract(self, text_list):
        """
        输入: 文本列表 ["Maths", "Number", ...]
        输出: Tensor [Batch, 768]
        """
        # 1. Tokenize
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=32, 
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # 2. Forward
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 方案 A: 取 [CLS] token (推荐)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            return cls_embeddings.cpu()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ==========================================
    # 1. 读取 JSON 数据并正确排序
    # ==========================================
    print(f"📂 正在读取知识点文件: {KNOWLEDGE_JSON}...")
    
    if not KNOWLEDGE_JSON.exists():
        raise FileNotFoundError(f"❌ 找不到文件: {KNOWLEDGE_JSON}")

    with open(KNOWLEDGE_JSON, 'r', encoding='utf-8') as f:
        know_map = json.load(f)
    
    # 🚨 关键步骤：按照 ID 的整数值排序
    # JSON 的 key 是字符串 ("0", "1", "10")。
    # 如果直接 sort，"10" 会排在 "2" 前面。
    # 所以必须用 key=lambda x: int(x)
    sorted_ids = sorted(know_map.keys(), key=lambda x: int(x))
    
    print(f"📊 检测到 {len(sorted_ids)} 个知识点")
    
    # 检查 ID 是否连续 (可选，防止中间缺 ID 导致行号错位)
    max_id = int(sorted_ids[-1])
    if max_id + 1 != len(sorted_ids):
        print(f"⚠️ 警告: ID 可能不连续！最大ID是 {max_id}, 但总数只有 {len(sorted_ids)}")
        # 如果你的模型依赖 embedding(id)，这通常意味着你需要填补空缺或重新映射
    
    # 生成对应的文本列表
    # texts[0] 就是 ID=0 的文本
    # texts[1] 就是 ID=1 的文本
    texts = [know_map[pid] for pid in sorted_ids]
    
    print(f"📝 样例检查:")
    print(f"   Row 0 (ID={sorted_ids[0]}): {texts[0]}")
    print(f"   Row 1 (ID={sorted_ids[1]}): {texts[1]}")
    # print(f"   Row 10 (ID={sorted_ids[10]}): {texts[10]}") # 如果有10的话

    # ==========================================
    # 2. 提取特征
    # ==========================================
    extractor = KnowledgeExtractor(MODEL_PATH, device)

    print("🚀 开始提取 BERT 特征...")
    # 因为知识点数量通常不多(几十几百个)，一次性提取最快
    emb_matrix = extractor.extract(texts)
    
    # Check 维度
    print(f"👀 提取结果形状: {emb_matrix.shape}") 
    # 应该是 [Total_Knowledge_Count, 768]

    # ==========================================
    # 3. 保存
    # ==========================================
    print(f"💾 保存到 {KNOW_OUTPUT_FILE}...")
    torch.save(emb_matrix, KNOW_OUTPUT_FILE)
    print("✅ 完成！")
    print(f"   现在你可以在模型中使用 nn.Embedding.from_pretrained(torch.load('{KNOW_OUTPUT_FILE}'))")

if __name__ == "__main__":
    main()
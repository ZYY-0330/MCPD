import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os
import json
import pandas as pd
from PIL import Image
import torch.nn.functional as F                  # <--- 用于 max_pool2d, interpolate
import torchvision.transforms.functional as TF   # <--- 用于 SquarePad 里的 pad
# 引入你的配置 (确保路径都在这里面定义好)
from configs.dataset_config import *
# 1. 定义填充类 (必须放在 RawDataDataset 外面或前面)
import torchvision.transforms.functional as TF  # <--- 1. 改名引用！

class SquarePad:
    def __call__(self, image):
        # 确保 image 是 PIL Image
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        padding = (p_left, p_top, max_wh - w - p_left, max_wh - h - p_top)
        
        # 2. 使用 TF.pad 而不是 F.pad
        return TF.pad(image, padding, 0, 'constant')

# 2. 修改后的 Dataset 类
class RawDataDataset(Dataset):
    def __init__(self, img_dir, text_dir, tokenizer, max_len=80):
        self.img_dir = img_dir 
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        print(f"    Loading text from {text_dir}...")
        with open(text_dir, 'r', encoding='utf-8') as f:
            self.text_data = json.load(f)
        self.pids = list(self.text_data.keys())
        
        # ✅✅✅ 修正后的 Transform ✅✅✅
        # 1. SquarePad: 保持比例，填充黑边 (解决长图变形问题)
        # 2. Resize: 缩放到 224x224
        # 3. ToTensor & Normalize: 标准化
        self.img_transform = transforms.Compose([
            SquarePad(),                   # <--- 核心修改！
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        base_dir = str(self.img_dir)
        img_path_jpg = os.path.join(base_dir, f"{pid}.jpg")
        img_path_png = os.path.join(base_dir, f"{pid}.png")
        
        img = None
        if os.path.exists(img_path_png):
            img = Image.open(img_path_png).convert('RGB')
        elif os.path.exists(img_path_jpg):
            img = Image.open(img_path_jpg).convert('RGB')
        
        if img is None:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            
        # 这里会调用上面定义好的含 SquarePad 的 transform
        img_tensor = self.img_transform(img)

       
     
        
        # --- B. 处理文本 ---
        # 根据你的json结构调整，假设 key 是 'content'
        if isinstance(self.text_data[pid], dict):
            content = self.text_data[pid].get('content', "")
        else:
            content = str(self.text_data[pid])

        encoding = self.tokenizer(
            content,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pid': int(pid),
            'image': img_tensor,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# ============================================================================
# 2. 离线模型定义 (冻结版)
# ============================================================================
class OfflineResNet(nn.Module):
    def __init__(self):
        super().__init__()
        print("    Loading ResNet50...")
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for param in self.parameters(): param.requires_grad = False
            
    def forward(self, x):
        x = self.layer0(x)
        l1 = self.layer1(x) 
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return [l1, l2, l3, l4]

class OfflineBERT(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        print(f"    Loading BERT from {model_path}...")
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.parameters(): param.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states

# ============================================================================
# 3. 主执行函数
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # 输出文件路径 (从config读取)
    # 确保 OUTPUT_FILE 在 dataset_config.py 里定义了，或者在这里直接写死路径
    # OUTPUT_FILE = "offline_features.pt" 
    
    # 1. 准备模型
    print("🚀 正在加载模型...")
    try:
        img_model = OfflineResNet().to(device).eval()
        txt_model = OfflineBERT(MODEL_PATH).to(device).eval()
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 准备数据
    print("📂 正在读取数据文件...")
    # 使用本文件定义的 RawDataDataset
    dataset = RawDataDataset(IMAGE_DIR, TEXT_DIR, tokenizer)
    
    total_items = len(dataset)
    print(f"📊 数据集统计: 共发现 {total_items} 道题目")
    
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    total_batches = len(loader)

    # 3. 开始提取
    print(f"🚀 开始特征提取 (Batch Size: {batch_size}, Total Batches: {total_batches})...")
    cached_data = {} 

    with torch.no_grad():
        pbar = tqdm(loader, total=total_batches, unit="batch", desc="Processing")
        
        for batch in pbar:
            pids = batch['pid'].tolist()
            imgs = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device) # [B, 80]

            # 跑模型
            img_feats = img_model(imgs) # list of 4 tensors
            txt_outputs = txt_model(input_ids, masks) 

            # 取出需要的层 (3, 8, 12)
            t_low = txt_outputs[3]
            t_mid = txt_outputs[8]
            t_high = txt_outputs[12]

           
                
                # --- 拆分 Batch 并转存 CPU ---
            for i, pid in enumerate(pids):
                # 1. 取出原始特征 (CPU)
                raw_l1 = img_feats[0][i].cpu() # 56x56
                raw_l2 = img_feats[1][i].cpu() # 28x28
                raw_l3 = img_feats[2][i].cpu() # 14x14
                raw_l4 = img_feats[3][i].cpu() # 7x7

                # =========================================================
                # 🚀 核心优化：形态学膨胀 + 自适应最大池化
                # =========================================================
                
                # A. 定义膨胀操作 (Dilation)
                # 原理：用 3x3 的最大值滤波 (stride=1, padding=1) 扫一遍
                # 效果：把 1px 的细线 "加粗" 到 3px，防止下采样时丢失
                # 注意：raw_l1 是 [C, H, W]，需要 unsqueeze(0) 变成 [1, C, H, W] 才能做 MaxPool
                
                l1_dilated = F.max_pool2d(raw_l1.unsqueeze(0), kernel_size=3, stride=1, padding=1)
                l2_dilated = F.max_pool2d(raw_l2.unsqueeze(0), kernel_size=3, stride=1, padding=1)
                
                # B. 执行最大池化下采样 (Downsampling)
                # 使用 adaptive_max_pool2d 强转 14x14
                # 相比 AvgPool，它只保留"有特征"的像素，不稀释信号
                l1_14 = F.adaptive_max_pool2d(l1_dilated, (14, 14)).squeeze(0)
                l2_14 = F.adaptive_max_pool2d(l2_dilated, (14, 14)).squeeze(0)
                
                # =========================================================

                # 2. Layer 3 (14x14) 不需要动，直接 Clone
                l3_14 = raw_l3.clone()
                
                # 3. Layer 4 (7x7) 太小，需要上采样 (插值)
                # 上采样只能用插值 (interpolate)，这里用双线性即可
                l4_14 = F.interpolate(raw_l4.unsqueeze(0), size=(14, 14), mode='bilinear', align_corners=True).squeeze(0)

                # 4. 打包图像特征
                i_data = [l1_14, l2_14, l3_14, l4_14]

                
                # 文本特征 (Low, Mid, High)
                t_data = [
                    t_low[i].cpu().clone(),
                    t_mid[i].cpu().clone(),
                    t_high[i].cpu().clone()
                ]
                
                # ✅ [关键] 保存 Mask!
                # 将 mask 转回 CPU 保存
                m_data = masks[i].cpu().clone()

                # 存入字典
                cached_data[pid] = {
                    "img": i_data,
                    "txt": t_data,
                    "mask": m_data  # <--- Mask在这里
                }
            
            pbar.set_description(f"Processing (Extracted: {len(cached_data)}/{total_items})")

    # 4. 保存
    print(f"\n💾 正在保存到 {OUTPUT_FILE} (这可能需要几秒钟)...")
    torch.save(cached_data, OUTPUT_FILE)
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"✅ 完成！文件大小: {file_size_mb:.2f} MB")
    print(f"✅ 成功提取了 {len(cached_data)} 道题目的特征。")

if __name__ == "__main__":
    main()
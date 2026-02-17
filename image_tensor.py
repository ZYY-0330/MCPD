# precompute_simple.py
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import time
from configs.dataset_config import *


# 和你训练时完全相同的transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    image_dir = Path(IMAGE_DIR)
    image_files = list(image_dir.glob("*.png"))  # 如果你的图像是jpg，改成 *.jpg
    
    print(f"找到 {len(image_files)} 个图像文件")
    print("开始预计算...")
    
    precomputed = {}
    start_time = time.time()
    
    for i, img_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"进度: {i}/{len(image_files)}")
        
        img = Image.open(img_path).convert('RGB')
        tensor_img = transform(img)
        precomputed[img_path.stem] = tensor_img
    
    # 保存
    torch.save(precomputed, IMAGE_OUTPUT_FILE)
    
    print(f"完成! 用时: {time.time()-start_time:.1f}秒")
    print(f"保存到: {IMAGE_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
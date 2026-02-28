# precompute_simple.py
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import time
from configs.dataset_config import *


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    image_dir = Path(IMAGE_DIR)
    image_files = list(image_dir.glob("*.png"))  
    precomputed = {}
    start_time = time.time()
    
    for i, img_path in enumerate(image_files):
        if i % 100 == 0:
 
        img = Image.open(img_path).convert('RGB')
        tensor_img = transform(img)
        precomputed[img_path.stem] = tensor_img
    
  
    torch.save(precomputed, IMAGE_OUTPUT_FILE)
    

if __name__ == "__main__":
    main()

import os
import time
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from NeuralNCDM import Net
import torch.multiprocessing as mp
from sklearn.metrics import f1_score
from torch.amp import autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch import dist, optim, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
from configs.dataset_config import *
from fusion_model import HierarchicalFusionSystem
from torch.nn.modules.module import _addindent
from dataset import RelationBuilder, RecordDataset
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from torch.utils.data import DistributedSampler



class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -np.inf

    def should_stop(self, current_metric):
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
class Trainer:
    def __init__(self, config, model, rank=0):
        self.config = config
        self.rank = rank
        self.model = model
        self._init_device()
        self._init_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1024)  
        self.best_metric = -float('inf')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(OUTPUT_DIR, 'NCDM_logs', f'exp_{timestamp}')  
        self.writer = SummaryWriter(log_dir=log_dir)  
        self.step_sum = 0
    def _set_requires_grad(self, params, requires_grad):
        for p in params:
            p.requires_grad = requires_grad
    def _init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.rank}')
            self.model.to(self.device)
            

            if dist.is_initialized():
                self.model = DDP(
                    self.model,
                    device_ids=[self.rank],
                    find_unused_parameters=True,  
                   
                    gradient_as_bucket_view=True  
                )
              
                               
        else:
            self.device = torch.device('cpu')
    def _init_optimizer(self):
       
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        LR_BASE  = 1e-3     
        LR_MODAL = 1e-4    
        WD_HIGH  = 1e-3         
        WD_LOW   = 1e-3     
        
      
        MODAL_PREFIXES = (
            'model_feat',     
            'know_projector',  
            'W_p',             
            'gate',           
            'diff_head',       
            'fusion'           
        ) 
        
        modal_params = [] 
        base_params = [] 
        
      

        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
       
            no_decay_list = ['bias', 'LayerNorm.weight']
            if any(nd in name for nd in no_decay_list):
                real_wd = 0.0
            else:
             
                if any(k in name for k in MODAL_PREFIXES):
                    real_wd = WD_HIGH 
                else:
                    real_wd = WD_LOW  

           
            if any(k in name for k in MODAL_PREFIXES):
                modal_params.append({
                    'params': param, 
                    'lr': LR_MODAL, 
                    'weight_decay': real_wd,
                    'name': name,
                    'initial_lr': LR_MODAL
                })
            else:
                base_params.append({
                    'params': param, 
                    'lr': LR_BASE, 
                    'weight_decay': real_wd, 
                    'name': name,
                    'initial_lr': LR_BASE
                })

     
        self.optimizer = torch.optim.AdamW(
            modal_params + base_params,
            lr=LR_BASE, 
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )

    def print_memory(self,tag=""):
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        print(f"[{tag}] allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
  
   
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        gcn_update = False

        all_targets = []
        all_preds = []
        all_probs = []

        progress_bar = tqdm(train_loader, 
                          desc=f"Epoch {epoch+1} [Rank {self.rank}]",
                          disable=not (self.rank == 0))
        

      
        WARMUP_STEPS = 50  
        INITIAL_LR_FACTOR = 1e-3 
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            for batch_idx, batch in enumerate(progress_bar):

                device = f'cuda:{self.rank}'
                new_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        new_batch[k] = v.to(device)
                    elif isinstance(v, list):
                        new_batch[k] = [x.to(device) for x in v if isinstance(x, torch.Tensor)]
                    else:
                        new_batch[k] = v
                batch = new_batch
                validate_device_consistency(batch, self.model)
                
                global_step = batch_idx + epoch * len(train_loader)
                
                if global_step < WARMUP_STEPS:
                    climbing_factor = (global_step + 1) / WARMUP_STEPS
                    
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        base_lr = param_group['initial_lr'] 
                        
                        start_lr = base_lr * INITIAL_LR_FACTOR

                        climbing_factor = (global_step + 1) / WARMUP_STEPS
                        
                        param_group['lr'] = start_lr + (base_lr - start_lr) * climbing_factor

                self.optimizer.zero_grad()
                with autocast(device_type='cuda',dtype=torch.float16):

                    output_1, pred_id, pred_img, alpha = self.model.forward(
                            batch
                        ) 
                    
                    targets = batch['corrects'].squeeze().float()
                        
                    output_fused = output_1.squeeze()
                       
                    loss_main = F.binary_cross_entropy_with_logits(output_fused, targets)

                        
                    loss = loss_main + 0.1*alpha
                        
                    probs = torch.sigmoid(output_fused) 
                    preds = (probs >= 0.5).long()  

                      
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)

                original_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                '''
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if "prednet_full" in name and "weight" in name:
                            param.clamp_(min=0.0)
                '''
                self.scaler.update()


    def train(self, train_loader, val_loader, test_loader, train_sampler=None, val_sampler=None):
        start_time = time.time()
        
        early_stopper = EarlyStopper(patience=5, min_delta=0.001)
       
        best_test_metrics = {
                'auc': {'value': 0, 'epoch': 0},
                'rmse': {'value': float('inf'), 'epoch': 0},
                'f1': {'value': 0, 'epoch': 0},
                'acc': {'value': 0, 'epoch': 0}
            }
        
        torch.autograd.set_detect_anomaly(True)

      
        TARGET_LR_MODAL = 1e-4 # 5e-5
        TARGET_LR_BASE  = 1e-3  # 1e-3
        TARGET_WD_MODAL = 1e-3
        TARGET_WD_BASE  = 1e-3

        if len(self.optimizer.param_groups) > 0:
            self.optimizer.param_groups[0]['lr'] = TARGET_LR_MODAL
            self.optimizer.param_groups[0]['initial_lr'] = TARGET_LR_MODAL
            self.optimizer.param_groups[0]['weight_decay'] = TARGET_WD_MODAL
            print(f"   >>> Group 0 Reset: LR={TARGET_LR_MODAL}, WD={TARGET_WD_MODAL}")

        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = TARGET_LR_BASE
            self.optimizer.param_groups[1]['initial_lr'] = TARGET_LR_BASE
            self.optimizer.param_groups[1]['weight_decay'] = TARGET_WD_BASE
            print(f"   >>> Group 1 Reset: LR={TARGET_LR_BASE},  WD={TARGET_WD_BASE}")
            
       
        for epoch in range(self.config['total_epochs']):
            
            if dist.is_initialized():
                if train_sampler is not None: train_sampler.set_epoch(epoch)
                if val_sampler is not None:   val_sampler.set_epoch(epoch)
            
           
            self.train_epoch(train_loader, epoch)

            val_metrics = self.validate(val_loader)
           
            if dist.is_initialized():
                sync_tensor = torch.tensor(0, device=self.device, dtype=torch.int32)
                dist.broadcast(sync_tensor, src=0)  # 阻塞直到所有进程到达此点

          
            test_metrics = self.validate(test_loader)
           
            if dist.is_initialized():
                dist.broadcast(torch.tensor([0], device=self.device), src=0)

        if dist.is_initialized():
            dist.destroy_process_group() 

           
        if self.rank == 0:
          

            best_log_str = (f"\nBest Test Results:\n"
                            f"- Best AUC: {best_test_metrics['auc']['value']:.6f} (Epoch {best_test_metrics['auc']['epoch']})\n"
                            f"- Best RMSE: {best_test_metrics['rmse']['value']:.6f} (Epoch {best_test_metrics['rmse']['epoch']})\n"
                            f"- Best F1: {best_test_metrics['f1']['value']:.6f} (Epoch {best_test_metrics['f1']['epoch']})\n"
                            f"- Best Acc: {best_test_metrics['acc']['value']:.6f} (Epoch {best_test_metrics['acc']['epoch']})\n"
                          

            print(best_log_str)
            print(f"训练完成，总耗时: {time.time() - start_time:.2f}秒")

           
       
    
  
    def validate(self, val_loader):
        self.model.eval()
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(f'cuda:{self.rank}') for k, v in batch.items()}
                with autocast(device_type='cuda'):
                    output_1 ,_,_,_= self.model.forward(
                       batch
                    ) 
                    output_1 =output_1.squeeze()

                    targets = (batch['corrects'].squeeze().float() >= 0.5).float()
                    probs = torch.sigmoid(output_1)
                   

                    
                all_targets.extend(targets.cpu().numpy().flatten())
                all_probs.extend(probs.detach().cpu().numpy().flatten())

        if dist.is_initialized():
            all_targets_tensor = torch.tensor(np.array(all_targets), dtype=torch.float, device=self.device)
            all_probs_tensor = torch.tensor(np.array(all_probs), device=self.device)
            target_list = [torch.zeros_like(all_targets_tensor) for _ in range(dist.get_world_size())]
            prob_list = [torch.zeros_like(all_probs_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(target_list, all_targets_tensor)
            dist.all_gather(prob_list, all_probs_tensor)
            all_targets = torch.cat(target_list).cpu().numpy()
            all_probs = torch.cat(prob_list).cpu().numpy()

        all_targets = np.array(all_targets)  
        all_targets = np.where(all_targets >= 0.5, 1.0, 0.0).astype(np.float32)

        all_probs = np.array(all_probs) 

        all_probs = np.clip(all_probs, 0.0, 1.0)

      
        total_samples = len(all_targets)
        all_preds = (all_probs >= 0.5).astype(int) 
        correct_predictions = (all_targets == all_preds).sum()
        
        epoch_acc = correct_predictions / total_samples
        epoch_auc = roc_auc_score(all_targets, all_probs)
        epoch_rmse = np.sqrt(mean_squared_error(all_targets, all_probs))
        epoch_f1 = f1_score(all_targets, all_preds)  

        return {
            'acc': epoch_acc, 
            'auc': epoch_auc, 
            'rmse': epoch_rmse,
            'f1': epoch_f1 
        }
   
   
image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

def validate_device_consistency(data, model):
    current_device = torch.cuda.current_device()
    # 检查数据设备
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            assert v.device == torch.device(f'cuda:{current_device}'), \
                f"数据 {k} 设备不一致: {v.device} vs cuda:{current_device}"
    # 检查模型设备
    for param in model.parameters():
        assert param.device == torch.device(f'cuda:{current_device}'), \
            f"模型参数设备不一致: {param.device}"

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size == 1 or not torch.cuda.is_available():
        return local_rank, world_size  
    
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=180),
            world_size=world_size,
            rank=rank
        )
    
    torch.cuda.set_device(local_rank)
    assert torch.cuda.current_device() == local_rank, \
        f"Device binding failed! Expected {local_rank}, got {torch.cuda.current_device()}"
    return local_rank, world_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=os.environ.get("LOCAL_RANK", 0))
    return parser.parse_args()
from EndToEndContrastiveModel import EndToEndContrastiveModel
if __name__ == "__main__":

    args = parse_args()
    local_rank, world_size = setup_distributed()
    torch.set_float32_matmul_precision('high')

    device = torch.device(f'cuda:{args.local_rank}' 
                         if torch.cuda.is_available() else 'cpu')
    
    torch.cuda.set_device(args.local_rank)  
   

    print(f"Rank {local_rank} 当前GPU: {torch.cuda.current_device()}")

    train_dataset = RecordDataset(mode='train',rank=args.local_rank)
    val_dataset = RecordDataset(mode='val',rank=args.local_rank)
    test_dataset = RecordDataset(mode='test',rank=args.local_rank)
    

    model = Net(
        student_n=train_dataset.user_n,
        exer_n=len(train_dataset.problem_data.valid_pids),
        knowledge_n=TOTAL_SKILLS,
        problem_dataset=train_dataset.problem_data
    ).to(device)
  
    config = {
        'total_epochs': 100,
       
    }

  
    trainer = Trainer(config, model, rank=local_rank)
    count_parameters(model)
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=local_rank)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=world_size, rank=local_rank)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, num_replicas=world_size, rank=local_rank)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
 
    train_loader = train_dataset.create_dataloader(train_sampler, 512, 0)
    val_loader = val_dataset.create_dataloader( val_sampler,512, 0)
    test_loader = test_dataset.create_dataloader( test_sampler, 512, 0)
    
    trainer.train(train_loader, val_loader, test_loader, train_sampler=train_sampler, val_sampler=val_sampler)

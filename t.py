import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 标准库
import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    # 1. 基本 Python 随机种子
    random.seed(seed)
    
    # 2. NumPy 随机种子
    np.random.seed(seed)
    
    # 3. PyTorch CPU 随机种子
    torch.manual_seed(seed)
    
    # 4. PyTorch GPU 随机种子 (针对当前显卡)
    torch.cuda.manual_seed(seed)
    
    # 5. PyTorch GPU 随机种子 (针对所有显卡，防止多卡训练不一致)
    torch.cuda.manual_seed_all(seed)
    
    # 6. 确定性算法配置 (关键：让 CuDNN 的运算结果也固定)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 7. 设置环境变量 (防止某些底层库产生随机性)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"随机种子已固定为: {seed}")

# 在主流程开始前调用
set_seed(42)
import os
import time
from datetime import datetime
import itertools
import h5py
from torch.optim.lr_scheduler import CosineAnnealingLR
from NeuralNCDM import Net
import torch.multiprocessing as mp
from sklearn.metrics import f1_score

from torch.amp import autocast
# 第三方库
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.cuda.amp import GradScaler
from torch import dist, optim, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from itertools import chain
# 本地模块
from dataset import ProblemDataset,DistributedBalancedProblemBatchSampler # 关键修改点1：替换数据集类
from BERT import MathBERTTextFeatureExtractor
from RestNet import FeatureExtractionModel
from configs.dataset_config import *
from fusion_model import HierarchicalFusionSystem
from torch.nn.modules.module import _addindent
from dataset import RelationBuilder, RecordDataset
from tqdm import tqdm
import h5py
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from torch.utils.data import DistributedSampler
import torch.nn.functional as F
from torch.utils.data import Subset
from UnifiedDataset import UnifiedDataset
import logging

import warnings
warnings.filterwarnings("ignore", message="adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation")


torch.autograd.set_detect_anomaly(True)
logging.basicConfig(
    filename='train.log',  # 指定日志文件
    level=logging.INFO,     # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1024)  # 混合精度梯度缩放
        self.best_metric = -float('inf')#记录最高分
        #self.loss_function = nn.CrossEntropyLoss()  # 交叉熵损失

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(OUTPUT_DIR, 'NCDM_logs', f'exp_{timestamp}')  
        self.writer = SummaryWriter(log_dir=log_dir)  
        self.step_sum = 0
    def _set_requires_grad(self, params, requires_grad):
        """批量设置参数梯度状态"""
        for p in params:
            p.requires_grad = requires_grad
    def _init_device(self):
        """设备初始化（支持多GPU）"""
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.rank}')
            self.model.to(self.device)
            

            if dist.is_initialized():
                self.model = DDP(
                    self.model,
                    device_ids=[self.rank],
                    find_unused_parameters=True,  # 必须启用
                   
                    gradient_as_bucket_view=True  # 提升效率
                )
              
                               
        else:
            self.device = torch.device('cpu')
    def _init_optimizer(self):
        """
        [Phase 2 Final Strategy] 双重分层策略 (兼容保存代码版)
        Group 0: Modal (LR=5e-5, WD=0.01)  -> 对应 TensorBoard LR_0
        Group 1: Base  (LR=1e-3, WD=1e-4)  -> 对应 TensorBoard LR_1
        """
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # ==========================================
        # 1. 定义超参
        # ==========================================
        LR_BASE  = 1e-3      # 基础学习率
        LR_MODAL = 1e-4    # 模态学习率 (5e-5)
        
        # 🌟 差异化权重衰减 (这是重点!)
        WD_HIGH  = 1e-3         # 强正则 (给模态)
        WD_LOW   = 1e-3      # 弱正则 (给基础)
        
        # ==========================================
        # 2. 定义前缀分组
        # ==========================================
        MODAL_PREFIXES = (
            'model_feat',      # 图像/文本融合层
            'know_projector',  # 知识点投影
            'W_p',             # Attention 参数
            'gate',            # 门控参数
            'diff_head',       # 难度预测头
            'fusion'           # 融合层
        ) 
        
        modal_params = [] # 对应 Group 0
        base_params = []  # 对应 Group 1
        
        print(f"\n⚡ [Optimizer] 初始化 Phase 2 双重分层模式...")
        print(f"   >>> Group 0 (Modal): LR={LR_MODAL}, WD={WD_HIGH}")
        print(f"   >>> Group 1 (Base) : LR={LR_BASE},  WD={WD_LOW}")

        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 1. 判断是否不衰减 (Bias/LayerNorm)
            no_decay_list = ['bias', 'LayerNorm.weight']
            if any(nd in name for nd in no_decay_list):
                real_wd = 0.0
            else:
                # 2. 如果不是 Bias，则根据组别决定 WD
                if any(k in name for k in MODAL_PREFIXES):
                    real_wd = WD_HIGH  # 模态组用 0.01
                else:
                    real_wd = WD_LOW   # 基础组用 0.0001

            # 3. 分组装填
            if any(k in name for k in MODAL_PREFIXES):
                modal_params.append({
                    'params': param, 
                    'lr': LR_MODAL, 
                    'weight_decay': real_wd, # ✅ 这里使用了差异化 WD
                    'name': name,
                    'initial_lr': LR_MODAL
                })
            else:
                base_params.append({
                    'params': param, 
                    'lr': LR_BASE, 
                    'weight_decay': real_wd, # ✅ 这里使用了差异化 WD
                    'name': name,
                    'initial_lr': LR_BASE
                })

        # ==========================================
        # 3. 初始化优化器
        # ==========================================
        # 注意顺序：modal_params 在前 (Group 0)，base_params 在后 (Group 1)
        # 这与你的 TensorBoard 记录代码完美对应
        self.optimizer = torch.optim.AdamW(
            modal_params + base_params,
            lr=LR_BASE, 
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )

        print(f"   >>> Group Modal Params: {len(modal_params)}")
        print(f"   >>> Group Base  Params: {len(base_params)}\n")
    '''
    def _init_optimizer(self):
        """
        初始化分层优化器 (Layer-wise LR)
        目标: Attention 跑慢 (1e-4)，NCDM 跑快 (1e-3)。
        """
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 1. 定义参数分组前缀
        FUSION_PREFIXES = ('model_feat','W_p', 'diff_head_k', 'know_pro',) 
        #'W_p', 'diff_head_k', 'know_pro'
        
        fusion_params = []   # 低速组 (1e-4)
        ncdm_params = []     # 高速组 (1e-3)
        

        FUSION_TARGET_LR = self.config['learning_rate_1'] # 0.0001
        NCDM_TARGET_LR = HIGH_LR = 0.0007

        # 2. 遍历参数，根据前缀分配速度
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 权重衰减分组 (标准操作)
            no_decay = ['bias', 'LayerNorm.weight']
            wd = self.config['weight_decay'] if not any(nd in name for nd in no_decay) else 0.0

            if name.startswith(FUSION_PREFIXES):
                # 融合核心，使用低学习率 (0.0001)
                fusion_params.append({
                    'params': param, 
                    'lr': FUSION_TARGET_LR, 
                    'weight_decay': wd, 
                    'name': name,
                    'initial_lr': FUSION_TARGET_LR # 关键：用明确的变量
                })
            else:
                # NCDM 核心 (Embeddings/Output)，使用高学习率 (0.0007)
                ncdm_params.append({
                    'params': param, 
                    'lr': NCDM_TARGET_LR, # 关键：使用明确的 NCDM 目标 LR 变量
                    'weight_decay': wd, 
                    'name': name,
                    'initial_lr': NCDM_TARGET_LR # 关键：用明确的变量
                })
        # 3. 初始化优化器 (将两个组都传入)
        self.optimizer = torch.optim.AdamW(
            fusion_params + ncdm_params,
            #lr=self.config['learning_rate_1'], # 这里的值不重要，因为我们对每个组都设置了 LR
            betas=(0.9, 0.999),
            eps=1e-8
        )
        if len(self.optimizer.param_groups) >= 2:
        # Group 0 (Fusion): 确保是 0.0001
            self.optimizer.param_groups[0]['lr'] = FUSION_TARGET_LR
            self.optimizer.param_groups[0]['initial_lr'] = FUSION_TARGET_LR # 重新赋值 initial_lr
            
            # Group 1 (NCDM): 强制设置为 0.0007
            self.optimizer.param_groups[1]['lr'] = NCDM_TARGET_LR 
            self.optimizer.param_groups[1]['initial_lr'] = NCDM_TARGET_LR # 重新赋值 initial_lr
            
        # 4. 调度器 (Plateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

        self.freeze_model_feat = False
        final_fusion_lr = self.optimizer.param_groups[0]['initial_lr']
        final_ncdm_lr = self.optimizer.param_groups[1]['initial_lr']
        print(f"--- Optimizer Init Check ---")
        print(f"Config LR 1 (Fusion Target): {self.config.get('learning_rate_1')}")
        print(f"NCDM Target HIGH_LR: {HIGH_LR}")
        print(f"Group 0 (Fusion) Target: {final_fusion_lr}")
        print(f"Group 1 (NCDM) Target: {final_ncdm_lr}")
        print(f"--- Check End ---")

    '''
    '''
    def _init_optimizer(self):
        """统一优化器配置，不分模块"""

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        # 所有参数都参与优化
        all_params = list(raw_model.named_parameters())

        # 权重衰减配置（bias和LayerNorm权重单独处理）
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            # 第一组：普通的权重 (Weight)，需要权重衰减 (防止过拟合)
            {
                'params': [p for n, p in all_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['weight_decay']  # 通常是0.01 或 0.05
            },
            # 第二组：偏置项 (Bias) 和 LayerNorm，不需要权重衰减 (这是业界标准做法)
            {
                'params': [p for n, p in all_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        # 初始化AdamW优化器
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate_1'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,    # 忍耐 5 个 Epoch 不涨再降
            verbose=True,
            min_lr=1e-6    # 最小不低于这个数
        )
        

        self.freeze_model_feat = False

    def _init_optimizer_without_model_feat(self):
        """冻结 model_feat，仅优化其他模块"""
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

       
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 只保留不属于 model_feat 的参数（已经冻结了）
        # 加上 module.
        all_params = [
            (n, p) for n, p in raw_model.named_parameters()
            if not n.startswith('module.model_feat')
        ]


        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in all_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['weight_decay']
            },
            {
                'params': [p for n, p in all_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate_2'],
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',    # 我们希望 AUC 越高越好
            factor=0.5,    # 涨不动了就 学习率 * 0.5
            patience=5,    # 忍耐 5 个 Epoch 不涨再降
            verbose=True,
            min_lr=1e-6    # 最小不低于这个数
        )

    '''

    def print_memory(self,tag=""):
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        print(f"[{tag}] allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
    def count_active_trainable_params(self,model, active_prefix='model_feat'):
        """
        计算并分割在当前 Batch 中获得了梯度的可训练参数量。
        :param model: DDP 或原始模型实例。
        :param active_prefix: 你的 Fusion 模块前缀 (例如 'model_feat')。
        """
        if not dist.is_initialized() or dist.get_rank() == 0:
            raw_model = model.module if hasattr(model, 'module') else model
            total_active_params = 0
            active_fusion_sum = 0
            active_ncdm_sum = 0
            
            NCDM_PREFIXES = ('student_emb', 'k_difficulty_NCDM', 'e_discrimination_NCDM', 'output_layer')
            
            for name, param in raw_model.named_parameters():
                # 1. 必须是可训练的参数 AND 2. 必须有梯度 (即被 forward/backward 流程用到)
                if param.requires_grad and param.grad is not None:
                    param_count = param.numel()
                    total_active_params += param_count
                    
                    # 区分是 NCDM 基础层还是 Fusion 层
                    if name.startswith(NCDM_PREFIXES):
                        active_ncdm_sum += param_count
                    elif name.startswith(active_prefix):
                        active_fusion_sum += param_count

            # 打印结果
            print("\n" + "="*80)
            print("💡 活跃参数量分割报告 (实际参与本 Batch 训练的参数)")
            print("="*80)
            print(f"总可训练参数 (Total Trainable): {sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}")
            print("-" * 80)
            print(f"1. 获得梯度的总参数 (Active): {total_active_params:,}")
            print(f"2. NCDM 基础参数 (Active): {active_ncdm_sum:,}")
            print(f"3. Fusion/Attention 系统 (Active): {active_fusion_sum:,}")
            print("-" * 80)
            print(f"   => 活跃的 Fusion 参数占总活跃参数的比例: {active_fusion_sum / total_active_params * 100:.2f}%")
            print("="*80)
    def _apply_phase_strategy(self, epoch):
        """
        [战略核心] 局部微调策略 (Partial Fine-tuning)
        
        策略逻辑：
        1. 全局原则：为了省显存，BERT/ResNet 的底层 (Bottom Layers) 永远锁死。
        2. 局部放开：BERT Layer 11 (顶层) 和 ResNet Layer 4 (顶层) 允许训练。
        3. Phase 1 (Epoch 0): 锁死 NCDM，强迫模态顶层和投影层学习。
        4. Phase 2 (Epoch 1+): 全员解冻 (除了底层骨干)。
        """
        # 定义阶段阈值 (只跑 1 个 epoch 热身足够了)
        PHASE_1_EPOCHS = 2
        
        # 1. 白名单：属于 Fusion 体系的组件
        FUSION_KEYWORDS = [
            'model_feat', 'diff_head', 'W_p', 'know_pro', 
            'output_layer', 'img_proj', 'text_proj', 'gate_weight','snr_diff_head',
        ]
        
        # 2. 深层冻结黑名单 (Deep Freeze List)
        # 这里的层永远不训练，用来省显存 + 保持基础特征
        DEEP_FREEZE_KEYWORDS = [
            # ResNet 底层 (锁 1, 2, 3 层; 放 layer4)
            'img_feature.backbone.conv1',
            'img_feature.backbone.bn1',
            'img_feature.backbone.layer1',
            'img_feature.backbone.layer2',
            'img_feature.backbone.layer3', 
            'img_feature.backbone.layer4', 
            # BERT 底层 (锁 0-10 层; 放 layer.11)
            'text_feature.bert_model.embeddings',
            'text_feature.bert_model.encoder.layer.0.',
            'text_feature.bert_model.encoder.layer.1.',
            'text_feature.bert_model.encoder.layer.2.',
            'text_feature.bert_model.encoder.layer.3.',
            'text_feature.bert_model.encoder.layer.4.',
            'text_feature.bert_model.encoder.layer.5.',
            'text_feature.bert_model.encoder.layer.6.',
            'text_feature.bert_model.encoder.layer.7.',
            'text_feature.bert_model.encoder.layer.8.',
            'text_feature.bert_model.encoder.layer.9.',
            'text_feature.bert_model.encoder.layer.10.', 
            'text_feature.bert_model.encoder.layer.11.', 
        ]

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        counts = {"frozen": 0, "training": 0}

        # --- 执行策略 ---
        for name, param in raw_model.named_parameters():
            # 默认：根据 Phase 决定是否开启
            should_train = True
            
            # 规则 A: Phase 1 只训模态
            if epoch < PHASE_1_EPOCHS:
                # 如果不是 Fusion 组件，就锁死 (锁 NCDM)
                if not any(k in name for k in FUSION_KEYWORDS):
                    should_train = False
            
            # 规则 B: 无论何时，底层骨干永远锁死 (一票否决)
            if any(k in name for k in DEEP_FREEZE_KEYWORDS):
                should_train = False
            
            # 应用设置
            param.requires_grad = should_train
            
            if should_train: counts["training"] += param.numel()
            else: counts["frozen"] += param.numel()

        # --- 仅主进程打印状态 ---
        if self.rank == 0:
            phase_name = "Phase 1: Modality Awakening (锁 NCDM)" if epoch < PHASE_1_EPOCHS else "Phase 2: Joint Optimization (全开)"
            print(f"\n{'='*60}")
            print(f"🚀 [Epoch {epoch}] 局部微调策略执行: {phase_name}")
            print(f"   >>> 训练参数量: {counts['training']:,} | 冻结参数量: {counts['frozen']:,}")
            print(f"{'='*60}")
            
            # 🔎 抽查关键层状态
            print("🔎 关键层抽查:")
            check_points = [
                ('BERT Bottom', 'text_feature.bert_model.encoder.layer.0.'), # 应锁
                ('BERT Top',    'text_feature.bert_model.encoder.layer.11.'),# Phase 1 应训
                ('ResNet Bott', 'img_feature.backbone.layer1'),              # 应锁
                ('ResNet Top',  'img_feature.backbone.layer4'),              # Phase 1 应训
                ('Diff Head',   'diff_head'),                                # Phase 1 应训
                ('Student Emb', 'student_emb')                               # Phase 1 锁, Phase 2 训
            ]
            for tag, key in check_points:
                found = False
                for name, param in raw_model.named_parameters():
                    if key in name:
                        status = "✅ 训练" if param.requires_grad else "🔒 冻结"
                        print(f"   - {tag:<12}: {status} ({name[:25]}...)")
                        found = True
                        break
                if not found: print(f"   - {tag:<12}: ⚠️ 未找到")
            print(f"{'='*60}\n")
    def train_epoch(self, train_loader, epoch):
        """单epoch训练（支持混合精度+分布式）"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        gcn_update = False

        # 新增：用于计算 RMSE 和 AUC
        all_targets = []
        all_preds = []
        all_probs = []

        # 进度条配置（仅主进程显示）
        progress_bar = tqdm(train_loader, 
                          desc=f"Epoch {epoch+1} [Rank {self.rank}]",
                          disable=not (self.rank == 0))
        

        # 定义 TensorBoard 日志存储目录
        
      


        # 🚨 Warmup 超参数设置 (解决 1e-4 启动失败的问题)
        WARMUP_STEPS = 50  # 假设前 500 个 Batch 进行热身
        INITIAL_LR_FACTOR = 1e-3 # 从 1e-3 * base_lr 开始
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            for batch_idx, batch in enumerate(progress_bar):
            
                
                # --- 兼容列表的数据搬运 ---
                device = f'cuda:{self.rank}'
                new_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        # 如果是 Tensor，直接搬运
                        new_batch[k] = v.to(device)
                    elif isinstance(v, list):
                        # ✅ 如果是列表 (比如 img_raw)，遍历列表里的每个 Tensor 搬运
                        new_batch[k] = [x.to(device) for x in v if isinstance(x, torch.Tensor)]
                    else:
                        # 其他类型 (如字符串) 保持原样
                        new_batch[k] = v
                batch = new_batch
                validate_device_consistency(batch, self.model)
                
                global_step = batch_idx + epoch * len(train_loader)
                
                if global_step < WARMUP_STEPS:
                    # 线性爬升因子：从 0 (或极小值) 爬升到 1.0
                    climbing_factor = (global_step + 1) / WARMUP_STEPS
                    
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        # 🚨 修正后的代码：使用保存的 'initial_lr' 作为基准
                        base_lr = param_group['initial_lr'] 
                        
                        # 计算当前步的 LR: 从 base_lr * 1e-3 爬升到 base_lr
                        start_lr = base_lr * INITIAL_LR_FACTOR
                        
                        # 线性爬升因子
                        climbing_factor = (global_step + 1) / WARMUP_STEPS
                        
                        # 更新 param_group['lr']
                        param_group['lr'] = start_lr + (base_lr - start_lr) * climbing_factor

                # 然后在您的训练步骤代码中，用 try-except 包裹整个步骤：
                try:
                
                    self.optimizer.zero_grad()
                    # 混合精度上下文
                    with autocast(device_type='cuda',dtype=torch.float16):
                        # --- [计时] Forward 开始 ---
                        t_fwd_start = time.time()
                        #self.print_memory("在forward前")
                        # 前向传播，获取模型输出
                        output_1, pred_id, pred_img, alpha = self.model.forward(
                            batch
                        ) 
                    
                        targets = batch['corrects'].squeeze().float()
                        
                        # 确保输出维度对齐 [Batch_Size]
                        output_fused = output_1.squeeze()
                        #output_id    = pred_id.squeeze()
                        #output_img   = pred_img.squeeze()

                        # 3. 计算三个独立的 BCE Loss
                        # (A) 主 Loss: 融合后的结果 (原本的 main_loss)
                        loss_main = F.binary_cross_entropy_with_logits(output_fused, targets)

                        # (B) ID 辅助 Loss: 强迫 ID 分支保持 0.78 的水准
                        #loss_id   = F.binary_cross_entropy_with_logits(output_id, targets)

                        # (C) 模态 辅助 Loss: 强迫图像分支必须自己学会预测！(这是重点)
                        #loss_img  = F.binary_cross_entropy_with_logits(output_img, targets)

                        # 4. 加权求和 (核心修改)
                        # 建议权重: 
                        # - main: 1.0 (主任务)
                        # - id:   0.5 (ID学得快，给小点权重即可)
                        # - img:  1.0 (图像学得慢，给大权重逼它学，替代你原来的 mse_loss)
                        loss = loss_main + 0.1*alpha
                        
                        # --- 正确计算概率 ---
                        probs = torch.sigmoid(output_fused)  # 形状 [batch_size]
                        preds = (probs >= 0.5).long()  # 二值化预测

                        #probs = output_1.squeeze(1)
                        #preds = (probs >= 0.5).long()  # shape [512]

                
                    # 反向传播
                    self.scaler.scale(loss).backward()
                

                    #if batch_idx == 0 and epoch == 0: # 只在第一轮的第一个 Batch 检查
                        # 调用上面的函数
                    #    self.count_active_trainable_params(self.model)
                    self.scaler.unscale_(self.optimizer)

                    if self.step_sum % 1000 == 0:
                        # 确保传入的是 self.model
                        self.comprehensive_gradient_analysis(self.model, self.scaler)
                    
                    self.step_sum += 1


                    original_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # 4. [记录] 记录各项指标 (只在主进程)
                    if self.rank == 0:
                        # 计算全局步数
                        global_step = batch_idx + epoch * len(progress_bar)
                        
                        # 记录 Loss
                        #self.writer.add_scalar('Batch/Total_Loss', loss.item(), global_step)
                        #self.writer.add_scalar('Batch/Main_Loss', loss_main.item(), global_step)
                        #self.writer.add_scalar('Batch/loss_id', loss_id.item(), global_step)
                        #self.writer.add_scalar('Batch/loss_img', loss_img.item(), global_step)
                        #self.writer.add_scalar('Batch/alpha', alpha.item(), global_step)
                       
                        
                        # 记录学习率
                        # 1. 获取各组当前 LR
                        # ----------------------------------------------------
                        # 组 0: 模态融合组 (Modal/Fusion)
                        lr_modal = self.optimizer.param_groups[0]['lr'] if len(self.optimizer.param_groups) > 0 else 0.0

                        # 组 1: NCDM 基础组 (Base/ID)
                        lr_base = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else 0.0

                        # ----------------------------------------------------
                        # 2. 记录到 TensorBoard
                        # ----------------------------------------------------

                        # 组 0: 模态融合组 (Modal/Fusion)
                        self.writer.add_scalar('Batch/LR_0_Modal_Fusion', lr_modal, global_step)

                        # 组 1: NCDM 基础组 (Base/ID)
                        self.writer.add_scalar('Batch/LR_1_Base_NCDM', lr_base, global_step)

                        # 记录梯度范数 (直接用 clip 返回的原始值，既准确又省了计算资源)
                        self.writer.add_scalar('Batch/Gradient_Norm', original_norm.item(), global_step)

                    self.scaler.step(self.optimizer)
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if "prednet_full" in name and "weight" in name:
                                # 强制将所有负权重直接归零，形成“硬墙”
                                param.clamp_(min=0.0)
                                #param.copy_(param.abs())

                    self.scaler.update()

                
                except Exception as e:
                    print(f"训练过程中出现异常: {e}")
                    import traceback
                    print("详细堆栈跟踪:")
                    print(traceback.format_exc())
                    
                    # 添加更多调试信息
                    print("\n=== 调试信息 ===")
                    print(f"Epoch: {epoch}, Batch index: {batch_idx}")
                    
                    # 检查输入数据
                    print("\n输入数据统计:")
                    for key, value in batch.items():
                        if torch.is_tensor(value):
                            print(f"{key}: shape={value.shape}, min={value.min().item()}, max={value.max().item()}, "
                                f"has_nan={torch.isnan(value).any().item()}, has_inf={torch.isinf(value).any().item()}")
                    
                    # 检查模型输出
                    if 'output_1' in locals():
                        print(f"\noutput_1: shape={output_1.shape}, min={output_1.min().item()}, max={output_1.max().item()}, "
                            f"has_nan={torch.isnan(output_1).any().item()}, has_inf={torch.isinf(output_1).any().item()}")
                    
                    if 'output' in locals():
                        print(f"output: shape={output.shape}, min={output.min().item()}, max={output.max().item()}, "
                            f"has_nan={torch.isnan(output).any().item()}, has_inf={torch.isinf(output).any().item()}")
                    
                    # 修复 mse_loss 的检查
                    if 'mse_loss' in locals() and mse_loss is not None:
                        if torch.is_tensor(mse_loss):
                            print(f"mse_loss: value={mse_loss.item()}, has_nan={torch.isnan(mse_loss).any().item()}, "
                                f"has_inf={torch.isinf(mse_loss).any().item()}")
                        else:
                            print(f"mse_loss: value={mse_loss} (not a tensor)")

                    # 修复 total_loss 的检查
                    if 'total_loss' in locals():
                        if torch.is_tensor(total_loss):
                            print(f"total_loss: value={total_loss.item()}, has_nan={torch.isnan(total_loss).any().item()}, "
                                f"has_inf={torch.isinf(total_loss).any().item()}")
                        else:
                            print(f"total_loss: value={total_loss} (not a tensor)")
                    
                    if 'main_loss' in locals():
                        print(f"main_loss: value={main_loss.item()}, has_nan={torch.isnan(main_loss).any().item()}, "
                            f"has_inf={torch.isinf(main_loss).any().item()}")

                    
                    # 检查模型参数
                    print("\n模型参数梯度统计:")
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_has_nan = torch.isnan(param.grad).any().item()
                            grad_has_inf = torch.isinf(param.grad).any().item()
                            print(f"{name}: grad_norm={grad_norm}, has_nan={grad_has_nan}, has_inf={grad_has_inf}")
                    
                    # 重新抛出异常以停止训练
                    raise


                
            
                # --- 累积指标 ---
                batch_loss = loss.item()
                batch_acc = (preds == targets.long()).float().mean().item()
                batch_rmse = np.sqrt(mean_squared_error(targets.cpu().numpy(), probs.detach().cpu().numpy()))
                batch_auc = roc_auc_score(targets.cpu().numpy(), probs.detach().cpu().numpy())

                total_loss += batch_loss * targets.size(0)
                total_samples += targets.size(0)
                correct_predictions += (preds == targets.long()).sum().item()
                all_targets.extend(targets.long().cpu().numpy().flatten())
                all_probs.extend(probs.detach().cpu().numpy().flatten())
                #self.print_memory("在计算all_targets和all_probs后")
                # --- 更新进度条 ---
            
                if self.rank == 0:
                    progress_bar.set_postfix({
                        'loss': f"{batch_loss:.6f}",
                        'acc': f"{batch_acc:.6f}",
                        'RMSE': f"{batch_rmse:.6f}",
                        'AUC': f"{batch_auc:.6f}"
                    })
               
        # --- 分布式同步（关键修改）---
        if dist.is_initialized():
            # 同步损失和准确率
            total_loss_tensor = torch.tensor(total_loss).to(self.device)
            total_samples_tensor = torch.tensor(total_samples).to(self.device)
            correct_tensor = torch.tensor(correct_predictions).to(self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()
            total_samples = total_samples_tensor.item()
            correct_predictions = correct_tensor.item()

            # 同步概率和标签
            all_targets_tensor = torch.tensor(np.array(all_targets), dtype=torch.long, device=self.device)
            all_probs_tensor = torch.tensor(np.array(all_probs), device=self.device)
            target_list = [torch.zeros_like(all_targets_tensor) for _ in range(dist.get_world_size())]
            prob_list = [torch.zeros_like(all_probs_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(target_list, all_targets_tensor)
            dist.all_gather(prob_list, all_probs_tensor)
            all_targets = torch.cat(target_list).cpu().numpy().astype(int)
            all_probs = torch.cat(prob_list).cpu().numpy()

        # --- 全局指标计算 ---
        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
        epoch_rmse = np.sqrt(mean_squared_error(all_targets, all_probs))
        epoch_auc = roc_auc_score(all_targets, all_probs)
        # 新增F1计算
        all_preds = (np.array(all_probs) >= 0.5).astype(int)
        epoch_f1 = f1_score(all_targets, all_preds)  # 二分类默认average='binary'
        
        if self.rank == 0:
            # 记录到TensorBoard
            
            self.writer.add_scalar('Epoch/Train_Loss', epoch_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', epoch_acc, epoch)
            self.writer.add_scalar('Epoch/Train_RMSE', epoch_rmse, epoch)
            self.writer.add_scalar('Epoch/Train_AUC', epoch_auc, epoch)
            self.writer.add_scalar('Epoch/Train_F1', epoch_f1, epoch)  # 新增行
        
        return epoch_loss, epoch_acc, epoch_rmse, epoch_auc, epoch_f1  # 返回F1
    '''
    def train(self, train_loader, val_loader,test_loader,train_sampler=None, val_sampler=None):
        """完整训练流程"""
        start_time = time.time()
        


        
        early_stopper = EarlyStopper(patience=5, min_delta=0.001)
       
        best_test_metrics = {
                'auc': {'value': 0, 'epoch': 0},
                'rmse': {'value': float('inf'), 'epoch': 0},  # RMSE 越小越好
                'f1': {'value': 0, 'epoch': 0},
                'acc': {'value': 0, 'epoch': 0}
            }
        
        # 在训练循环开始之前（比如在 Trainer 类的初始化方法或训练开始的方法中）添加：
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.config['total_epochs']):
            # 在每个 epoch 开始时设置 sampler 的 epoch,保证多显卡训练时，每一轮数据都能真随机打乱
            if dist.is_initialized():
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)  # 训练集需要 shuffle
                if val_sampler is not None:
                    val_sampler.set_epoch(epoch)  # 验证集如果需要 shuffle 也设置
            if epoch == 100 and not self.freeze_model_feat:
                #print("🔄 Epoch 1 reached: Switching to phase 2 (freeze model_feat)")
                
                # 1⃣️ 冻结参数
                for name, param in self.model.named_parameters():
                    if name.startswith('module.model_feat.'):
                        param.requires_grad = False

                #print("参数冻结完毕，检测一下requires_grad：")
               
                torch.distributed.barrier()#多显卡同步 (Barrier)
                
                # 2⃣️ 重新构建优化器
                self._init_optimizer_without_model_feat()
                #print("新优化器参数数量：", sum(p.numel() for group in self.optimizer.param_groups for p in group['params']))
                
                self.freeze_model_feat = True
                
                torch.distributed.barrier()



            # 训练阶段
            train_loss, train_acc, train_rmse, train_auc,train_f1 = self.train_epoch(train_loader, epoch)

            

            val_metrics = self.validate(val_loader)
        '''
    def train(self, train_loader, val_loader, test_loader, train_sampler=None, val_sampler=None):
        """完整训练流程 (已集成 Phase 1/2 策略)"""
        start_time = time.time()
        
        early_stopper = EarlyStopper(patience=5, min_delta=0.001)
       
        best_test_metrics = {
                'auc': {'value': 0, 'epoch': 0},
                'rmse': {'value': float('inf'), 'epoch': 0},
                'f1': {'value': 0, 'epoch': 0},
                'acc': {'value': 0, 'epoch': 0}
            }
        
        torch.autograd.set_detect_anomaly(True)

        print("\n🔧 [Manual Fix] 强制重置学习率与权重衰减...")
        
        TARGET_LR_MODAL = 1e-4 # 5e-5
        TARGET_LR_BASE  = 1e-3  # 1e-3
        TARGET_WD_MODAL = 1e-3
        TARGET_WD_BASE  = 1e-3

        # Group 0: Modal
        if len(self.optimizer.param_groups) > 0:
            self.optimizer.param_groups[0]['lr'] = TARGET_LR_MODAL
            self.optimizer.param_groups[0]['initial_lr'] = TARGET_LR_MODAL
            self.optimizer.param_groups[0]['weight_decay'] = TARGET_WD_MODAL
            print(f"   >>> Group 0 Reset: LR={TARGET_LR_MODAL}, WD={TARGET_WD_MODAL}")

        # Group 1: Base
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = TARGET_LR_BASE
            self.optimizer.param_groups[1]['initial_lr'] = TARGET_LR_BASE
            self.optimizer.param_groups[1]['weight_decay'] = TARGET_WD_BASE
            print(f"   >>> Group 1 Reset: LR={TARGET_LR_BASE},  WD={TARGET_WD_BASE}")
            
        print("✅ 重置完成，开始训练循环...\n")
        # ==================== 训练主循环 ====================
        for epoch in range(self.config['total_epochs']):
            
            # 1. 设置 Sampler (多卡必备)
            if dist.is_initialized():
                if train_sampler is not None: train_sampler.set_epoch(epoch)
                if val_sampler is not None:   val_sampler.set_epoch(epoch)
            
            # 2. 🔥【关键修改】调用分阶段策略
            # 这行代码会自动根据 epoch 决定锁死哪些层
            #self._apply_phase_strategy(epoch)
            
            # 注意：不需要 barrier，因为所有显卡都会运行这行代码
            
            # 3. 训练阶段
            train_loss, train_acc, train_rmse, train_auc, train_f1 = self.train_epoch(train_loader, epoch)

            # 4. 验证阶段
            val_metrics = self.validate(val_loader)
            # 验证阶段（仅主进程）
            if self.rank == 0:
                
                self.writer.add_scalar('Epoch/Val_AUC', val_metrics['auc'], epoch)
                self.writer.add_scalar('Epoch/Val_RMSE', val_metrics['rmse'], epoch)
                self.writer.add_scalar('Epoch/Val_Accuracy', val_metrics['acc'], epoch)
                self.writer.add_scalar('Epoch/Val_F1', val_metrics['f1'], epoch)
                
                if val_metrics['auc'] > self.best_metric:
                    self.best_metric = val_metrics['auc']
                    #self._save_checkpoint(epoch, val_metrics)

                log_str = (f"Epoch {epoch+1}/{self.config['total_epochs']} | "
                            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | Train RMSE: {train_rmse:.6f} | Train F1: {train_f1:.6f} | Train AUC: {train_auc:.6f} | "
                            f"Val AUC: {val_metrics['auc']:.6f} | Val RMSE: {val_metrics['rmse']:.6f} | Val F1: {val_metrics['f1']:.6f} |Val Acc: {val_metrics['acc']:.6f}")
                logging.info(log_str)
                print(log_str)
                self.scheduler.step(val_metrics['auc'])
                with open("train_logs_E.txt", "a") as f:
                    f.write(log_str + "\n")

                if early_stopper.should_stop(val_metrics['auc']):
                    print(f"Early stopping at epoch {epoch}")
                    break

            # 同步点：所有进程必须参与
            if dist.is_initialized():
                # 创建同步张量（必须相同设备和类型）
                sync_tensor = torch.tensor(0, device=self.device, dtype=torch.int32)
                dist.broadcast(sync_tensor, src=0)  # 阻塞直到所有进程到达此点

            # 记录最佳测试结果，分别存储4个指标的最佳值及其对应的 epoch
            # 仅主进程测试
            test_metrics = self.validate(test_loader)
            if epoch % 1 == 0 and self.rank == 0:
                
                self.writer.add_scalar('Epoch/Test_AUC', test_metrics['auc'], epoch)
                self.writer.add_scalar('Epoch/Test_RMSE', test_metrics['rmse'], epoch)
                self.writer.add_scalar('Epoch/Test_Accuracy', test_metrics['acc'], epoch)
                self.writer.add_scalar('Epoch/Test_F1', test_metrics['f1'], epoch)

                log_str = (f"Epoch {epoch+1}/{self.config['total_epochs']} | "
                            f"Test AUC: {test_metrics['auc']:.6f} | Test RMSE: {test_metrics['rmse']:.6f} | "
                            f"Test F1: {test_metrics['f1']:.6f} | Test Acc: {test_metrics['acc']:.6f}")
                
                logging.info(log_str)
                print(log_str)
                with open("train_logs_E.txt", "a") as f:
                    f.write(log_str + "\n")

                # 记录最佳测试结果（分别判断每个指标是否更优）
                if test_metrics['auc'] > best_test_metrics['auc']['value']:
                    best_test_metrics['auc']['value'] = test_metrics['auc']
                    best_test_metrics['auc']['epoch'] = epoch + 1

                if test_metrics['rmse'] < best_test_metrics['rmse']['value']:  # RMSE 越小越好
                    best_test_metrics['rmse']['value'] = test_metrics['rmse']
                    best_test_metrics['rmse']['epoch'] = epoch + 1

                if test_metrics['f1'] > best_test_metrics['f1']['value']:
                    best_test_metrics['f1']['value'] = test_metrics['f1']
                    best_test_metrics['f1']['epoch'] = epoch + 1

                if test_metrics['acc'] > best_test_metrics['acc']['value']:
                    best_test_metrics['acc']['value'] = test_metrics['acc']
                    best_test_metrics['acc']['epoch'] = epoch + 1

            # 确保所有进程都同步
            if dist.is_initialized():
                dist.broadcast(torch.tensor([0], device=self.device), src=0)

            # 在训练循环结束后添加
        if dist.is_initialized():
            dist.destroy_process_group()  # 必须的清理操作 

            # 主进程保存最终模型并打印最佳结果
        if self.rank == 0:
            #self._save_checkpoint(epoch, val_metrics, final=True)

            # Get current learning rate from the optimizer
            current_lr = self.optimizer.param_groups[0]['lr']
            init_lr = self.config['learning_rate_1']  # 💡 加上这一句获取初始学习率

            best_log_str = (f"\nBest Test Results:\n"
                            f"- Best AUC: {best_test_metrics['auc']['value']:.6f} (Epoch {best_test_metrics['auc']['epoch']})\n"
                            f"- Best RMSE: {best_test_metrics['rmse']['value']:.6f} (Epoch {best_test_metrics['rmse']['epoch']})\n"
                            f"- Best F1: {best_test_metrics['f1']['value']:.6f} (Epoch {best_test_metrics['f1']['epoch']})\n"
                            f"- Best Acc: {best_test_metrics['acc']['value']:.6f} (Epoch {best_test_metrics['acc']['epoch']})\n"
                            f"- Initial Learning Rate: {init_lr:.8f}\n"      # ✨ 新增这一行
                            f"- Current Learning Rate: {current_lr:.8f}\n") # 原来这一行保留

            print(best_log_str)
            logging.info(best_log_str)
            with open("train_logs_E.txt", "a") as f:
                f.write(best_log_str)

            print(f"训练完成，总耗时: {time.time() - start_time:.2f}秒")

           
       
    
    def _save_checkpoint(self, epoch, metrics, final=False):
        """模型保存（含元数据）"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ncdm_{'final' if final else 'best'}_{timestamp}_auc{metrics['auc']:.6f}.pt"
        save_path = os.path.join(self.config['model_dir'], filename)
        
        torch.save(checkpoint, save_path)
        print(f"模型已保存至: {save_path}")

    def print_gpu(self,tag=""):
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
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

                    # 关键修改1：显式二值化标签
                    targets = (batch['corrects'].squeeze().float() >= 0.5).float()
                    probs = torch.sigmoid(output_1)
                    if torch.isnan(output_1).any():
                        print("🔥 output_1 里出现 NaN")
                    if torch.isnan(probs).any():
                        print("🔥 probs 里出现 NaN")

                    
                all_targets.extend(targets.cpu().numpy().flatten())
                all_probs.extend(probs.detach().cpu().numpy().flatten())

        # --- 分布式同步 ---
        if dist.is_initialized():
            all_targets_tensor = torch.tensor(np.array(all_targets), dtype=torch.float, device=self.device)
            all_probs_tensor = torch.tensor(np.array(all_probs), device=self.device)
            target_list = [torch.zeros_like(all_targets_tensor) for _ in range(dist.get_world_size())]
            prob_list = [torch.zeros_like(all_probs_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(target_list, all_targets_tensor)
            dist.all_gather(prob_list, all_probs_tensor)
            all_targets = torch.cat(target_list).cpu().numpy()
            all_probs = torch.cat(prob_list).cpu().numpy()

        # 关键修改2：同步后再次二值化
        all_targets = np.array(all_targets)  # 转换列表为数组
        all_targets = np.where(all_targets >= 0.5, 1.0, 0.0).astype(np.float32)

            # 关键修改3：确保 all_probs 是数组
        # --- 确保 all_probs 是数组 ---
        all_probs = np.array(all_probs)  # 转换列表为数组

        # --- 修正：确保 all_probs 在 [0, 1] 之间 ---
        all_probs = np.clip(all_probs, 0.0, 1.0)

        # --- 合法性检查 ---
        assert np.isin(all_targets, [0, 1]).all(), f"非法标签值: {np.unique(all_targets)}"
        assert (all_probs >= 0).all() and (all_probs <= 1).all(), f"概率值超出范围: {np.unique(all_probs)}"


        # --- 指标计算 ---
        # --- 指标计算 ---
        total_samples = len(all_targets)
        all_preds = (all_probs >= 0.5).astype(int)  # 新增预测标签生成
        correct_predictions = (all_targets == all_preds).sum()
        
        epoch_acc = correct_predictions / total_samples
        epoch_auc = roc_auc_score(all_targets, all_probs)
        epoch_rmse = np.sqrt(mean_squared_error(all_targets, all_probs))
        epoch_f1 = f1_score(all_targets, all_preds)  # 新增F1计算

        return {
            'acc': epoch_acc, 
            'auc': epoch_auc, 
            'rmse': epoch_rmse,
            'f1': epoch_f1  # 新增返回项
        }
    '''
    def comprehensive_gradient_analysis(self, model, scaler):
        # 分布式检查：只在主进程执行，节省其他显卡的资源
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # ================= 配置区 =================
        THRESHOLD_WEIGHT_SMALL = 1e-6   
        THRESHOLD_BIAS_SMALL = 1e-9    
        
        print("\n" + "="*80)
        print("🔬 智能梯度健康检查报告 (DDP 兼容版)")
        print("="*80)
        
        total_norm = 0.0
        vanished_weights = 0
        total_weights = 0
        total_biases = 0
        has_nan = False
        
        module_stats = {}

        for name, p in model.named_parameters():
            if p.grad is None:
                continue

            # 【新增】关键安全检查：检测 NaN 和 Inf
            # 这一步非常重要，因为一旦出现 NaN，后续的 norm 计算都会失效
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                has_nan = True
                print(f"💀 [致命错误] 在层 {name} 中发现 NaN/Inf！")
                # 遇到 NaN 通常可以直接跳过统计，或者记录下来
                continue

            grad_norm = p.grad.detach().norm().item()
            total_norm += grad_norm ** 2
            
            # 区分 Bias 和 Weight
            is_bias = 'bias' in name.lower() or 'norm' in name.lower()
            
            if is_bias:
                total_biases += 1
                if grad_norm < THRESHOLD_BIAS_SMALL:
                    # Bias 消失通常不需要报警
                    pass
            else:
                total_weights += 1
                if grad_norm < THRESHOLD_WEIGHT_SMALL:
                    vanished_weights += 1
                    # 限制打印数量，防止刷屏
                    if vanished_weights <= 5:
                        print(f"🔴 [权重死寂] {name}: {grad_norm:.2e}")

            # =====================================================
            # 【关键修复】 DDP 命名前缀处理
            # =====================================================
            # 如果是 DDP，名字是 "module.backbone.0..."
            # 我们需要去掉 "module." 才能提取真正的模块名 "backbone"
            clean_name = name
            if clean_name.startswith('module.'):
                clean_name = clean_name[7:] # 去掉前7个字符 "module."
            
            # 提取第一级模块名
            module_name = clean_name.split('.')[0]
            
            if module_name not in module_stats:
                module_stats[module_name] = {'grad_sum': 0.0, 'count': 0, 'max': 0.0}
            
            module_stats[module_name]['grad_sum'] += grad_norm
            module_stats[module_name]['count'] += 1
            module_stats[module_name]['max'] = max(module_stats[module_name]['max'], grad_norm)

        total_norm = total_norm ** 0.5
        
        # ================= 输出摘要 =================
        print("-" * 80)
        print(f"📊 整体健康度摘要:")
        
        if has_nan:
            print(f"   💀 状态: 【危险】检测到 NaN 或 Inf，模型可能已发散！")
        else:
            print(f"   ➤ 总梯度范数: {total_norm:.4f}")
        
        print(f"   ➤ 当前 Scale: {scaler.get_scale()}")
        
        w_vanish_rate = (vanished_weights/total_weights*100) if total_weights > 0 else 0
        print(f"   ➤ 权重层活跃度: {total_weights - vanished_weights}/{total_weights} (消失率: {w_vanish_rate:.1f}%)")
        
        if w_vanish_rate > 20:
             print(f"      🔴 警告：大量权重停止更新！")

        # ================= 模块透视 =================
        print("\n🛠️  各模块“出力”情况 (平均梯度):")
        print(f"   {'模块名':<25} | {'平均梯度':<12} | {'最大梯度':<12} | {'状态'}")
        print("-" * 80)
        
        # 排序输出，方便查看
        for name in sorted(module_stats.keys()):
            stats = module_stats[name]
            avg_grad = stats['grad_sum'] / stats['count']
            
            status = ""
            if avg_grad > 1.0: status = "💣 可能爆炸"
            elif avg_grad > 1e-2: status = "🔥 剧烈更新"
            elif avg_grad > 1e-3: status = "✅ 稳步更新"
            elif avg_grad > 1e-5: status = "💤 微调中"
            else: status = "❄️ 几乎冻结"
            
            print(f"   {name:<25} | {avg_grad:.2e}     | {stats['max']:.2e}     | {status}")

        print("="*80 + "\n")

    '''
    def comprehensive_gradient_analysis(self, model, scaler):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        print("\n" + "="*80)
        print("🔬 梯度健康检查 - 所有参数")
        print("="*80)
        
        # 解包模型
        if hasattr(model, 'module'):
            real_model = model.module
        else:
            real_model = model
        
        if hasattr(real_model, '_orig_mod'):
            real_model = real_model._orig_mod
        
        # 首先检查冻结参数
        frozen_params = []
        trainable_params = []
        
        for name, param in real_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        # 输出冻结参数信息
        if frozen_params:
            print("🧊 冻结参数统计:")
            print(f"  共有 {len(frozen_params)} 个参数被冻结 (requires_grad=False)")
            
            # 按模块分组统计冻结参数
            frozen_by_module = {}
            for name in frozen_params:
                parts = name.split('.')
                module_name = parts[0] if parts else 'unknown'
                if len(parts) > 1:
                    module_name = f"{parts[0]}.{parts[1]}"
                
                if module_name not in frozen_by_module:
                    frozen_by_module[module_name] = []
                frozen_by_module[module_name].append(name)
            
            print("\n📌 按模块冻结参数统计:")
            for module, params in frozen_by_module.items():
                print(f"  📍 {module}: {len(params)} 个冻结参数")
                # 显示前3个冻结参数作为示例
                for i, param_name in enumerate(params[:3]):
                    print(f"     {i+1}. {param_name}")
                if len(params) > 3:
                    print(f"     ... 还有 {len(params)-3} 个参数")
            
            print("-"*80 + "\n")
        
        # 只对可训练参数进行梯度分析
        print("📋 可训练参数的梯度分析:")
        print("-"*100)
        
        # 收集所有可训练参数的梯度信息
        grad_info = []
        has_nan = False
        total_norm = 0.0
        
        for name, param in real_model.named_parameters():
            if not param.requires_grad:
                continue  # 跳过冻结参数
                
            if param.grad is None:
                grad_norm = 0.0
                has_grad = False
            else:
                grad = param.grad
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    has_nan = True
                    grad_norm = float('nan')
                    has_grad = True
                else:
                    grad_norm = grad.norm().item()
                    total_norm += grad_norm ** 2
                    has_grad = True
            
            # 存储信息
            grad_info.append({
                'name': name,
                'grad_norm': grad_norm,
                'has_grad': has_grad,
                'has_nan': torch.isnan(grad).any() if has_grad else False,
                'has_inf': torch.isinf(grad).any() if has_grad else False
            })
            
            # 显示每个参数的梯度
            if has_grad and not (torch.isnan(grad).any() or torch.isinf(grad).any()):
                # 按梯度大小排序显示
                status = ""
                if grad_norm > 1.0: status = "💣"
                elif grad_norm > 0.1: status = "🔥"
                elif grad_norm > 0.01: status = "✅"
                elif grad_norm > 0.001: status = "💤"
                else: status = "❄️"
                
                print(f"{status} {name}: grad_norm={grad_norm:.6f}")
        
        # 计算总梯度范数
        total_norm = total_norm ** 0.5 if total_norm > 0 else 0.0
        
        # 排序显示
        if grad_info:
            print("\n📊 梯度最大前10个参数:")
            sorted_grads = sorted([g for g in grad_info if g['has_grad'] and not g['has_nan'] and not g['has_inf']], 
                                key=lambda x: x['grad_norm'], reverse=True)
            
            for i, g in enumerate(sorted_grads[:10]):
                print(f"{i+1:2d}. {g['name']}: {g['grad_norm']:.6f}")
            
            print("\n📉 梯度最小前10个参数:")
            sorted_grads_small = sorted([g for g in grad_info if g['has_grad'] and not g['has_nan'] and not g['has_inf']], 
                                    key=lambda x: x['grad_norm'])
            
            for i, g in enumerate(sorted_grads_small[:10]):
                print(f"{i+1:2d}. {g['name']}: {g['grad_norm']:.6f}")
        
        print("\n" + "-"*80)
        print(f"📈 汇总:")
        print(f"   总可训练参数: {len(trainable_params)}")
        if frozen_params:
            print(f"   冻结参数: {len(frozen_params)}")
        print(f"   总梯度范数: {total_norm:.6f}")
        if grad_info:
            print(f"   有梯度参数: {sum(1 for g in grad_info if g['has_grad'])}")
            print(f"   零梯度参数: {sum(1 for g in grad_info if not g['has_grad'])}")
        if has_nan:
            print(f"   💀 检测到NaN/Inf梯度!")
        print("="*80 + "\n")

# 配置图像预处理（与ProblemDataset一致）
image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])
torch.autograd.set_detect_anomaly(True)  # 加入这行！运行后会显示具体报错位置


import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from filelock import FileLock
import os
def save_fused_features(model, dataset, batch_size=4):
    """
    保存融合特征、文本特征和图像特征为 .pt 文件
    :param model: 模型
    :param dataset: 数据集
    :param output_dir: 输出目录
    :param batch_size: 批大小
    """
    # 确保进程组已初始化
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://"
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 如果是 DDP 包装的模型，获取原始模型
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    # 将模型移动到当前设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 数据加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda batch: {
            'pids': [item['pid'] for item in batch],
            'image': torch.stack([item['image'] for item in batch]),
            'text': [item['text'] for item in batch]
        }
    )

    # 创建输出目录（仅在主进程中）
    if rank == 0:
        os.makedirs(TEXT_FEATURES_DIR, exist_ok=True)
        os.makedirs(IMAGE_FEATURES_DIR, exist_ok=True)
        os.makedirs(FUSION_FEATURES_PATH, exist_ok=True)
    dist.barrier()  # 同步所有进程



    with torch.no_grad():  # 禁用梯度计算
        gate_dict = {}

        for batch in tqdm(loader, desc="特征提取与保存", disable=rank != 0):
            batch = {
                'pid': batch['pids'],
                'image': batch['image'].to(device),
                'text': batch['text']
            }

            text_feat = model.text_feature.process_batch(
                list(zip(batch['pid'], batch['text']))
            )
            text_feat = [x.to(device) for x in text_feat]
            #img_feats, img_feat = model.img_feature(batch['image'].float().to(device))
            #fused_feat = model.extract_features(batch)
            

            for i, pid in enumerate(batch['pid']):
                '''
                # ======================= 融合特征保存 =======================
                file_path = FUSION_FEATURES_PATH / f"{pid}.pt"
                lock_path = str(file_path) + ".lock"
                with FileLock(lock_path):
                    if file_path.exists():
                        #print(f"文件已存在: {file_path}")
                        os.remove(file_path)
                    torch.save(fused_feat[i].cpu(), file_path)
                '''
                 # ======================= 文本特征保存（只保存最后一层） =======================
                file_path = TEXT_FEATURES_DIR / f"{pid}.pt"
                lock_path = str(file_path) + ".lock"
                with FileLock(lock_path):
                    if file_path.exists():
                        os.remove(file_path)
                    torch.save(text_feat[-1][i].cpu(), file_path)  # 这里text_feat[-1]是最后一层，索引i取对应样本
                '''
                # ======================= 图像特征保存 =======================
                file_path = IMAGE_FEATURES_DIR / f"{pid}.pt"
                lock_path = str(file_path) + ".lock"
                with FileLock(lock_path):
                    if file_path.exists():
                        print(f"文件已存在: {file_path}")
                        os.remove(file_path)
                    torch.save(img_feat[i].cpu(), file_path)
                '''

        # 同步所有进程
        dist.barrier()
def save_Img_Text_features(model, dataset, batch_size=4):
    """
    保存融合特征、文本特征和图像特征为 .pt 文件
    :param model: 模型
    :param dataset: 数据集
    :param output_dir: 输出目录
    :param batch_size: 批大小
    """
    # 确保进程组已初始化
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://"
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 如果是 DDP 包装的模型，获取原始模型
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    # 将模型移动到当前设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 数据加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda batch: {
            'pids': [item['pid'] for item in batch],
            'image': torch.stack([item['image'] for item in batch]),
            'text': [item['text'] for item in batch]
        }
    )

    with torch.no_grad():  # 禁用梯度计算
        gate_dict = {}

        for batch in tqdm(loader, desc="特征提取与保存", disable=rank != 0):
            batch = {
                'pid': batch['pids'],
                'image': batch['image'].to(device),
                'text': batch['text']
            }

            model.text_feature.save_features(list(zip(batch['pid'], batch['text'])))
            
            model.img_feature.save_features_from_images(batch['image'], batch['pid'])
            
        # 同步所有进程
        dist.barrier()


import torch.distributed as dist

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
        

import os
import torch
import torch.distributed as dist
from datetime import timedelta
import argparse
from torch.utils.data import DataLoader, DistributedSampler
# 在 __main__ 块最前面添加
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 强制与nvidia-smi设备顺序一致 
os.environ["NCCL_IB_DISABLE"] = "1"             # 禁用InfiniBand（因日志显示mlx5设备未找到）

def setup_distributed():
    # 自动检测运行模式
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 单卡模式直接返回
    if world_size == 1 or not torch.cuda.is_available():
        return local_rank, world_size  # 返回local_rank而不是全局rank
    
    # 多卡初始化流程
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=180),
            world_size=world_size,
            rank=rank
        )
    
    # 显式设备绑定（核心修改）
    torch.cuda.set_device(local_rank)
    # 验证设备绑定
    assert torch.cuda.current_device() == local_rank, \
        f"Device binding failed! Expected {local_rank}, got {torch.cuda.current_device()}"
    return local_rank, world_size

def parse_args():
    parser = argparse.ArgumentParser()
    # 必须保留的参数（torchrun会自动注入）
    parser.add_argument("--local_rank", type=int, default=os.environ.get("LOCAL_RANK", 0))
    # 其他参数...
    return parser.parse_args()
from EndToEndContrastiveModel import EndToEndContrastiveModel
if __name__ == "__main__":

    # 初始化分布式环境
    args = parse_args()
    local_rank, world_size = setup_distributed()
    torch.set_float32_matmul_precision('high')

   


          

    
    
    # 调试信息
    print(f"[Process {os.getpid()}] "
          f"Local Rank: {args.local_rank}, "
          f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}, "
          f"Current Device: {torch.cuda.current_device()}")
    
    # 设备初始化
    device = torch.device(f'cuda:{args.local_rank}' 
                         if torch.cuda.is_available() else 'cpu')
    
    torch.cuda.set_device(args.local_rank)  # 显式强制绑定设备
   

    # 打印设备信息（调试用）
    print(f"Rank {local_rank} 当前GPU: {torch.cuda.current_device()}")
    # 初始化数据集
    train_dataset = RecordDataset(mode='train',rank=args.local_rank)
    val_dataset = RecordDataset(mode='val',rank=args.local_rank)
    test_dataset = RecordDataset(mode='test',rank=args.local_rank)
    

    print("1")
    #problem_dataset = ProblemDataset(transform=image_transform)
    print("2")
    # 初始化模型
    model = Net(
        student_n=train_dataset.user_n,
        exer_n=len(train_dataset.problem_data.valid_pids),
        knowledge_n=TOTAL_SKILLS,
        problem_dataset=train_dataset.problem_data
    ).to(device)
    
    '''
    print(f"Rank {args.local_rank}: Compiling model...")
    try:
        model = torch.compile(model, mode='default')
    except Exception as e:
        print(f"编译失败，回退: {e}")
    '''
    '''
    print(f"Rank {args.local_rank}: Compiling model...")
    try:
        # 启用动态形状支持
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.assume_static_by_default = False
        
        # 编译模型，允许动态形状
        model = torch.compile(
            model, 
            mode='default',
            dynamic=True,  # 启用动态形状
            fullgraph=False,
        )
        print("✅ 模型编译成功（动态形状模式）")
    except Exception as e:
        print(f"⚠️ 编译失败，回退到未编译模式: {e}")
    '''
    
    try:
        # 强制重置
        torch._dynamo.reset()
        
        model = torch.compile(
            model,
            mode="reduce-overhead", # 或者 default
            dynamic=True,
            backend="inductor",
            # 关键在这里：传入 options 字典
            options={
                "shape_padding": True,  # 强制填充动态维度，避免 symbolic计算错误
                "triton.cudagraphs": False, # DDP 下 cudagraphs 极易报错，建议关闭
            }
        )
        print("✅ 模型编译成功 (开启 Shape Padding)")
    except Exception as e:
        print(f"编译失败: {e}")
    print("3")
    # 初始化配置
    config = {
        'total_epochs': 100,
        'learning_rate_1': 0.0001,
        'learning_rate_2': 0.0001,
        'weight_decay': 0.01,
        'grad_clip': 0.5,
        'log_interval': 50,
        'model_dir': '/mnt/proj/autodl-tmp/checkpoints',
        'use_amp': True
    }

    print("4")

    trainer = Trainer(config, model, rank=local_rank)
    print("5")
    def count_parameters(model):
        """
        计算并打印 NCDM 基础层和 Fusion 层的参数量分割。
        :param model: 你的主模型实例。
        :param user_params: 从你的代码中获取的配置参数 (例如学生数，知识点数)。
        """
        if not dist.is_initialized() or dist.get_rank() == 0:
            total_trainable = 0
            ncdm_core_sum = 0
            fusion_system_sum = 0
            
            # 识别关键参数组
            NCDM_CORE_PREFIXES = ('student_emb', 'k_difficulty_NCDM', 'e_discrimination_NCDM', 'output_layer','W_p', 'diff_head_k', 'know_pro')
            FUSION_CORE_PREFIXES = ('model_feat')
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                param_count = param.numel()
                total_trainable += param_count
                
                # 检查是否属于 NCDM 基础层
                if name.startswith(NCDM_CORE_PREFIXES):
                    ncdm_core_sum += param_count
                # 检查是否属于 Fusion/Attention 层 (model_feat 是你的融合主模块)
                elif name.startswith(FUSION_CORE_PREFIXES):
                    fusion_system_sum += param_count
                else:
                    # 剩下的参数，通常是 DDP 的包装或未命名的参数
                    pass

            # 打印详细分割报告
            print("\n" + "="*80)
            print("💡 核心模块参数量分割报告 (可训练部分)")
            print("="*80)
            print(f"总可训练参数 (Total Trainable): {total_trainable:,}")
            print("-" * 80)
            print(f"1. NCDM 基础参数 (Embeddings/Heads): {ncdm_core_sum:,}")
            print(f"2. Fusion/Attention 系统 (model_feat): {fusion_system_sum:,}")
            print(f"3. 剩余参数 (如DDP包装/未识别): {total_trainable - ncdm_core_sum - fusion_system_sum:,}")
            print("-" * 80)
            print(f"   => Fusion 系统占总可训练参数的比例: {fusion_system_sum / total_trainable * 100:.2f}%")
            print("="*80)

    # 在训练开始前调用
    count_parameters(model)
    print("6")
    '''
    train_dataset.records = train_dataset.records[:3000]
    val_dataset.records = val_dataset.records[:3000]
    test_dataset.records = test_dataset.records[:3000]
    '''
    

    '''
    batch_size = 1024
    max_problems = 557

    train_sampler = DistributedBalancedProblemBatchSampler(
        train_dataset,
        batch_size=batch_size,
        max_problems=max_problems,
        num_replicas=world_size,
        rank=local_rank,
        seed=42
    )
    val_sampler = DistributedBalancedProblemBatchSampler(
        val_dataset,
        batch_size=batch_size,
        max_problems=max_problems,
        num_replicas=world_size,
        rank=local_rank,
        seed=42
    )
    test_sampler = DistributedBalancedProblemBatchSampler(
        test_dataset,
        batch_size=batch_size,
        max_problems=max_problems,
        num_replicas=world_size,
        rank=local_rank,
        seed=42
    )
    '''
    # 数据加载器配置（分布式）
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=local_rank)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=world_size, rank=local_rank)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, num_replicas=world_size, rank=local_rank)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    # 关键调试点：打印各进程数据索引范围
    if dist.is_initialized() and local_rank == 0:
        print(f"Rank {local_rank} 训练集索引示例: {list(train_sampler)[:5]}")
        print(f"Rank {local_rank} 验证集索引示例: {list(val_sampler)[:5]}")

    print("7")

    # 配置 DataLoader
    
    '''
    # 创建DataLoader注意这里用batch_sampler
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # 用batch_sampler替代sampler + batch_size
        num_workers=14,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=14,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=14,
        pin_memory=True,    
        collate_fn=test_dataset.collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )
    '''
    train_loader = train_dataset.create_dataloader(train_sampler, 512, 0)
    val_loader = val_dataset.create_dataloader( val_sampler,512, 0)
    test_loader = test_dataset.create_dataloader( test_sampler, 512, 0)
    
    # 启动训练
    trainer.train(train_loader, val_loader, test_loader, train_sampler=train_sampler, val_sampler=val_sampler)
    print("9")

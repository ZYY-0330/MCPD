import torch
import torch.nn as nn
from  BERT import MathBERTTextFeatureExtractor
from  RestNet import FeatureExtractionModel
from  fusion_model import HierarchicalFusionSystem
import time
import gc
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

class EndToEndContrastiveModel(nn.Module):
    """整合所有组件的端到端模型"""
    def __init__(self, device='auto'):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == 'auto' else device
        )


        # 文本特征提取器
        self.text_feature = MathBERTTextFeatureExtractor()
        
        # 图像特征提取器 (使用预训练的ResNet-50)
        self.img_feature = FeatureExtractionModel().float().to(self.device)
        
        # 多模态融合模块
        self.fusion = HierarchicalFusionSystem().float().to(self.device)
        
        # 先定义线性投影
        self.img_proj = nn.Linear(2048, 1024)  # 把图像通道投成768

        self.text_proj = nn.Linear(768, 1024)

        # 为对比学习添加投影层 (可选，也可以在forward中动态创建)
        self.text_projector_0 = nn.Linear(768, 256)
        self.text_projector_1 = nn.Linear(768, 256) 
        self.text_projector_2 = nn.Linear(768, 256)


        # 特征适配器
        self.feature_adapter = nn.Sequential(
            nn.Conv1d(768, 512, kernel_size=1),  # 从768维到512个通道
            nn.AdaptiveMaxPool1d(50)  # 池化到固定维度50
        )

        nn.init.xavier_uniform_(self.img_proj.weight)  # 顶层初始化
        if self.img_proj.bias is not None:
            nn.init.constant_(self.img_proj.bias, 0)

        self.to(self.device)
    def contrastive_loss_multi_scale(self, img_feats_list, text_feats_list, temperature=0.01):
        def check_tensor(name, tensor):
            if torch.isnan(tensor).any():
                print(f"❌ 警告：{name} 中出现 NaN ❌")
            else:
                print(f"✅ {name} 正常：均值={tensor.mean().item():.4f}, 标准差={tensor.std().item():.4f}, 范数={tensor.norm().item():.4f}")
                max_val = tensor.max().item()
                min_val = tensor.min().item()
                print(f"   数值范围: [{min_val:.6f}, {max_val:.6f}]")
                mean_val = tensor.mean().item()

                # 检查是否数值过大或过小
                if abs(mean_val) > 100 or abs(max_val) > 1000:
                    print(f"⚠️  注意：{name} 数值范围可能过大")
                if abs(mean_val) < 1e-8 and abs(max_val) < 1e-6:
                    print(f"⚠️  注意：{name} 数值过小，可能梯度消失")

        for i in range(len(img_feats_list)):
            check_tensor(f"刚送入对比学习loss的图像尺度{i} img_feats", img_feats_list[i])
        for i in range(len(text_feats_list)):
            check_tensor(f"刚送入对比学习loss的文本尺度{i} text_feats", text_feats_list[i])
        total_loss = 0
        num_scales = len(img_feats_list)
        text_projectors = [self.text_projector_0, self.text_projector_1, self.text_projector_2]
        
        for i, (img_feat, text_feat, projector) in enumerate(zip(img_feats_list, text_feats_list, text_projectors)):
            # 图像特征处理
            img_pooled = img_feat.mean(dim=[2, 3])
            
            # 文本特征处理
            text_pooled = text_feat.mean(dim=1)
            text_pooled = projector(text_pooled)
            
            # L2归一化
            img_pooled = F.normalize(img_pooled, p=2, dim=1)
            text_pooled = F.normalize(text_pooled, p=2, dim=1)
            
            # ========== 添加诊断代码 ==========
            print(f"\n=== 尺度 {i} 对比学习诊断 ===")
            print(f"对比学习loss 图像特征 - 形状: {img_pooled.shape}, 均值: {img_pooled.mean():.6f}, 标准差: {img_pooled.std():.6f}")
            print(f"对比学习loss 文本特征 - 形状: {text_pooled.shape}, 均值: {text_pooled.mean():.6f}, 标准差: {text_pooled.std():.6f}")
            
            # 计算相似度
            similarity = torch.matmul(img_pooled, text_pooled.t()) / temperature
            
            print(f"对比学习loss 相似度矩阵范围: [{similarity.min():.6f}, {similarity.max():.6f}]")
            print(f"对比学习loss温度参数: {temperature}")
            
            # 检查正负样本相似度
            batch_size = img_pooled.size(0)
            positive_sim = similarity.diag()  # 对角线是正样本
            negative_sim = similarity[~torch.eye(batch_size, dtype=bool, device=similarity.device)]  # 非对角线是负样本
            
            print(f"对比学习loss 正样本相似度: 均值={positive_sim.mean():.6f}, 范围=[{positive_sim.min():.6f}, {positive_sim.max():.6f}]")
            print(f"对比学习loss 负样本相似度: 均值={negative_sim.mean():.6f}, 范围=[{negative_sim.min():.6f}, {negative_sim.max():.6f}]")
            print(f"对比学习loss 相似度差异(正-负): {(positive_sim.mean() - negative_sim.mean()):.6f}")
            
            # 计算准确率
            labels = torch.arange(batch_size).to(img_pooled.device)
            img2txt_pred = similarity.argmax(dim=1)
            txt2img_pred = similarity.argmax(dim=0)
            img2txt_acc = (img2txt_pred == labels).float().mean()
            txt2img_acc = (txt2img_pred == labels).float().mean()
            
            print(f"对比学习loss 图像→文本检索准确率: {img2txt_acc.item():.4f}")
            print(f"对比学习loss 文本→图像检索准确率: {txt2img_acc.item():.4f}")
            # ========== 诊断代码结束 ==========
            
            # 计算对比loss
            loss_i2t = F.cross_entropy(similarity, labels)
            loss_t2i = F.cross_entropy(similarity.t(), labels)
            scale_loss = (loss_i2t + loss_t2i) / 2
            
            total_loss += scale_loss
            print(f"对比学习loss 尺度 {i} 对比loss: {scale_loss.item():.4f}")
        
        return total_loss / num_scales
    '''
    def contrastive_loss_fused_features(self, fused_features, temperature=0.1):
        """
        用融合后的特征计算对比loss，促进题目间区分
        fused_features: [batch_size, feature_dim] 融合后的题目表示
        temperature: 温度参数，控制分布尖锐程度
        """
        batch_size = fused_features.shape[0]
        
        # 1. L2归一化
        fused_features = F.normalize(fused_features, p=2, dim=1)
        
        # 2. 计算相似度矩阵
        similarity = torch.matmul(fused_features, fused_features.t()) / temperature
        # similarity形状: [batch_size, batch_size]
        
        # 3. 创建标签 - 每个样本的正样本是自己
        labels = torch.arange(batch_size).to(fused_features.device)
        
        # 4. 计算对比损失
        loss = F.cross_entropy(similarity, labels)
        
        # 5. 添加诊断信息
        print(f"\n=== 融合特征对比学习诊断 ===")
        print(f"融合特征 - 形状: {fused_features.shape}, 均值: {fused_features.mean():.6f}, 标准差: {fused_features.std():.6f}")
        print(f"相似度矩阵范围: [{similarity.min():.6f}, {similarity.max():.6f}]")
        
        # 正样本相似度（对角线）
        positive_sim = similarity.diag()
        # 负样本相似度（非对角线）
        negative_mask = ~torch.eye(batch_size, dtype=bool, device=similarity.device)
        negative_sim = similarity[negative_mask]
        
        print(f"正样本相似度: 均值={positive_sim.mean():.6f}")
        print(f"负样本相似度: 均值={negative_sim.mean():.6f}")
        print(f"相似度差异(正-负): {(positive_sim.mean() - negative_sim.mean()):.6f}")
        
        # 计算准确率
        pred = similarity.argmax(dim=1)
        accuracy = (pred == labels).float().mean()
       
        print(f"检索准确率: {accuracy.item():.4f}")
        print(f"融合特征对比loss: {loss.item():.4f}")
       
        return loss
    '''
    def contrastive_loss_fused_features(self,fused_features, temperature=0.1):
        """
        计算对比损失 (contrastive loss)
        
        参数:
            fused_features: torch.Tensor, [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
            temperature: float, 温度系数
            method: str, 'mean' 或 'flatten'
                - 'mean': 对序列维度求平均，得到 [batch_size, feature_dim]
                - 'flatten': 展开所有序列元素，得到 [batch_size*seq_len, feature_dim]

        返回:
            loss: 对比损失
        """
        method = 'mean'
        # 如果是三维张量
        if fused_features.dim() == 3:
            if method == 'mean':
                # 对序列维度求平均
                fused_features = fused_features.mean(dim=1)  # [batch_size, feature_dim]
            elif method == 'flatten':
                batch_size, seq_len, feature_dim = fused_features.shape
                fused_features = fused_features.view(batch_size * seq_len, feature_dim)
            else:
                raise ValueError("method must be 'mean' or 'flatten'")
        
        # L2 归一化
        fused_features = F.normalize(fused_features, p=2, dim=1)  # [N, feature_dim]

        # 计算相似度矩阵
        similarity = torch.matmul(fused_features, fused_features.t()) / temperature  # [N, N]

        # 标签：每个样本与自己为正样本
        labels = torch.arange(fused_features.size(0), device=fused_features.device)

        # 对比损失
        loss = F.cross_entropy(similarity, labels)

        return loss
    def forward(self, img_raw_list, txt_raw_list,padding_mask):

       

        
       
        img_feats= self.img_feature(img_raw_list)
       
        fused_feat, img_rep, txt_rep = self.fusion(txt_raw_list, img_feats, padding_mask)  # 直接传入mask
        #end_block(t0, "3. Cross-Modal Fusion", self.device)
        
        #print("****text_feats****")
        #print("text_feats:", [f.shape for f in text_feats])
        #print("**************")
        # 多模态融合函数定义
        '''
        def fusion_forward(*args):
            text_feats_input = args[:3]
            image_feats_input = args[3:]
            return self.fusion(text_feats_input, image_feats_input)

        # 准备输入（这里仅保留结构一致）
        text_inputs = tuple(text_feats)
        image_inputs = tuple(img_feats)

        # 多模态融合（直接调用，不使用 checkpoint）
        t_fusion_start = time.time()
        fused_feat = fusion_forward(*text_inputs, *image_inputs)
        t_fusion_done = time.time()
        #print(f"[计时] 多模态融合耗时: {t_fusion_done - t_fusion_start:.4f}s")
        '''
        '''
        if torch.cuda.current_device() == 0:
            print("****多模态融合****")
            self.print_memory("在多模态融合后")
        '''
        # 总耗时
        #print(f"[计时] forward 总耗时: {time.time() - t_start:.4f}s")
        '''
        loss = 0
        for t_feat, i_feat in zip(text_feats, img_feats):
            # 文本做池化 (B, 80, 768) -> (B, 768)
            if t_feat.dim() == 3:
                t_feat = t_feat.mean(dim=1)
            # 图像做池化 (B, C, H, W) -> (B, C)
            if i_feat.dim() == 4:
                i_feat = i_feat.mean(dim=[2, 3])
            # 图像特征通道投影到768
            i_feat = self.img_proj(i_feat)  # (B, 768)

            loss += 1 - F.cosine_similarity(t_feat, i_feat, dim=1).mean()  # 最小化余弦距离
        loss = loss / len(text_feats)
        '''
        '''
        loss = 0
        for t_feat, i_feat in zip(text_feats, img_feats):
            # 文本做池化 (B, 80, 768) -> (B, 768)
            if t_feat.dim() == 3:
                t_feat = t_feat.mean(dim=1)

            # 图像特征 (B, 49, 1024) -> (B, 1024)
            if i_feat.dim() == 3:
                i_feat = i_feat.mean(dim=1)

            # 线性投影到768维
            i_feat = self.img_proj(i_feat)  # (B, 768)

            # 计算余弦距离
            loss += 1 - F.cosine_similarity(t_feat, i_feat, dim=1).mean()

        loss = loss / len(text_feats)
        '''
        #contrast_loss = self.contrastive_loss_multi_scale(img_feats, text_feats, temperature=0.1)
        contrast_loss = self.contrastive_loss_fused_features(fused_feat, temperature=0.1)
        #contrast_loss = torch.tensor(0.0)
        return  fused_feat, img_rep, txt_rep

    '''
    def forward(self, batch):
        device = next(self.parameters()).device  # 自动拿模型所在设备

        # 把 batch 中所有 Tensor 移动到相同设备
        t_start = time.time()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        t_move_done = time.time()
        #print(f"[计时] batch数据搬移耗时: {t_move_done - t_start:.4f}s")

        # 图像特征提取
        t_img_start = time.time()
        img_feats, img_feat = self.img_feature(batch['image'])
        t_img_done = time.time()
        #print(f"[计时] 图像特征提取耗时: {t_img_done - t_img_start:.4f}s")

        # 文本特征提取
        t_text_start = time.time()
        text_feats = self.text_feature.process_batch(
            list(zip(batch['pid'], batch['text']))
        )
        t_text_done = time.time()
        #print(f"[计时] 文本特征提取耗时: {t_text_done - t_text_start:.4f}s")

        # 多模态融合函数定义
        def fusion_forward(*args):
            text_feats_input = args[:3]
            image_feats_input = args[3:]
            return self.fusion(text_feats_input, image_feats_input)

        # 准备 checkpoint 输入
        text_inputs = tuple(text_feats)
        image_inputs = tuple(img_feats)

        # 多模态融合
        t_fusion_start = time.time()
        fused_feat = checkpoint.checkpoint(fusion_forward, *text_inputs, *image_inputs, use_reentrant=False)
        t_fusion_done = time.time()
        #print(f"[计时] 多模态融合耗时: {t_fusion_done - t_fusion_start:.4f}s")

        # 总耗时
        #print(f"[计时] forward 总耗时: {time.time() - t_start:.4f}s")

        loss = 0
        for t_feat, i_feat in zip(text_feats, img_feats):
            # 文本做池化 (B, 80, 768) -> (B, 768)
            if t_feat.dim() == 3:
                t_feat = t_feat.mean(dim=1)
            # 图像做池化 (B, C, H, W) -> (B, C)
            if i_feat.dim() == 4:
                i_feat = i_feat.mean(dim=[2,3])
            # 图像特征通道投影到768
            i_feat = self.img_proj(i_feat)  # (B, 768)
            
            loss += F.mse_loss(t_feat, i_feat)
        loss = loss / len(text_feats)



        return None, loss, fused_feat
    '''

    def print_memory(self,tag=""):
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        print(f"[{tag}] allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")

    def extract_features(self, batch):
        """
        专用特征提取方法，跳过负样本处理流程
        参数：
            batch: 仅需包含 {'pid', 'image', 'text'}
        返回：
            融合特征张量 [B, feature_dim]
        """
        # 文本特征提取
        text_feat = self.text_feature.process_batch(
            list(zip(batch['pid'], batch['text']))
        )
        text_feat = [x.to(self.device) for x in text_feat]
        
        # 图像特征提取（优化张量转换逻辑）
        images = batch['image'].float().to(self.device)
        img_feats,img_feat = self.img_feature(images)  # [B, 512, 56, 56]
        
        # 直接融合特征
        fused_feat = self.fusion(text_feat, img_feats)
        
        # 统一输出格式
        return fused_feat
    
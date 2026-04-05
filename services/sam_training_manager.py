# SAM训练管理器（茎秆+雄穗专用优化版 - 框选Prompt + IndexError修复）
import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


# 配置参数（茎秆+雄穗专用优化）
class Config:
    SAM_MODEL_TYPE = "vit_b"  # 优先用vit_b，3060无压力
    SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"  # 对应vit_b的权重
    INPUT_SIZE = 1024
    BATCH_SIZE = 1  # 3060显存限制
    LEARNING_RATE = 3e-5  # 稍低的初始学习率，保护边缘特征
    EPOCHS = 30  # 建议30轮
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # SAM官方归一化参数（ImageNet均值标准差）
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # 茎秆+雄穗专用优化参数
    MASK_DILATE_KERNEL = 5  # 掩码膨胀核大小
    MASK_DILATE_ITER = 2   # 掩码膨胀次数
    POS_WEIGHT = 250       # 极端前景权重
    GRADIENT_ACCUMULATION_STEPS = 6  # 梯度累积步数，等效batch=6


class SingleStemTasselDataset(Dataset):
    """茎秆+雄穗专用SAM训练数据集（框选Prompt版）"""

    def __init__(self, coco_container, image_paths):
        self.samples = self._prepare_single_instance_samples(coco_container, image_paths)
        self.input_size = Config.INPUT_SIZE

    @staticmethod
    def _read_image_rgb(image_path):
        """稳健读取图片，兼容 Windows 中文路径和 OpenCV 读图失败。"""
        if not image_path or not os.path.exists(image_path):
            return None

        image = cv2.imread(image_path)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            buffer = np.fromfile(image_path, dtype=np.uint8)
            if buffer.size > 0:
                decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if decoded is not None:
                    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        try:
            with Image.open(image_path) as pil_image:
                return np.array(pil_image.convert("RGB"))
        except Exception:
            return None

    def _prepare_single_instance_samples(self, coco_container, image_paths):
        """将COCO多实例标注拆分为单株茎秆+雄穗样本"""
        samples = []
        for image_path in image_paths:
            if image_path not in coco_container:
                continue

            annotation = coco_container[image_path]
            if not annotation.get("image_state", {}).get("annotation_completed", False):
                continue

            # 读取原始图像尺寸
            orig_img = self._read_image_rgb(image_path)
            if orig_img is None:
                print(f"跳过无法读取的训练图片: {image_path}")
                continue
            orig_h, orig_w = orig_img.shape[:2]

            # 遍历每一株玉米（单实例）
            plants = annotation.get("plants", [])
            for plant in plants:
                if "polygons" not in plant or len(plant["polygons"]) == 0:
                    continue

                # 合并该植株的所有多边形（仅茎秆+雄穗）
                all_points = []
                for poly in plant["polygons"]:
                    if len(poly) >= 3:
                        all_points.extend(poly)

                if len(all_points) < 3:
                    continue

                # 计算该植株的边界框（作为训练用的框选Prompt）
                all_points_np = np.array(all_points, dtype=np.float32)
                x_min, y_min = np.min(all_points_np, axis=0)
                x_max, y_max = np.max(all_points_np, axis=0)

                # 稍微扩展边界框（更符合SAM提示习惯）
                box_w = x_max - x_min
                box_h = y_max - y_min
                x_min = max(0, x_min - box_w * 0.1)
                y_min = max(0, y_min - box_h * 0.1)
                x_max = min(orig_w, x_max + box_w * 0.1)
                y_max = min(orig_h, y_max + box_h * 0.1)

                samples.append({
                    "image_path": image_path,
                    "orig_size": (orig_h, orig_w),
                    "plant_polygons": plant["polygons"],  # 仅茎秆+雄穗的多边形
                    "bbox": [x_min, y_min, x_max, y_max]  # 框选Prompt
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        orig_h, orig_w = sample["orig_size"]
        plant_polygons = sample["plant_polygons"]
        bbox = sample["bbox"]

        # 1. 加载并预处理图像（严格遵循SAM官方流程）
        image = self._read_image_rgb(image_path)
        if image is None:
            raise ValueError(f"无法读取训练图片: {image_path}")

        # 2. 生成单实例掩码（仅茎秆+雄穗）
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for poly in plant_polygons:
            points = np.array(poly, dtype=np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 255)

        # 3. 【核心优化】对茎秆+雄穗掩码做膨胀强化
        # 先膨胀，强化细长前景信号
        kernel = np.ones((Config.MASK_DILATE_KERNEL, Config.MASK_DILATE_KERNEL), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=Config.MASK_DILATE_ITER)

        # 4. 统一resize到SAM输入尺寸
        scale = self.input_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        image_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        bbox_resized = np.array(bbox) * scale

        # 5. 填充到1024x1024（保持长宽比）
        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w

        image_padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        image_padded[:new_h, :new_w, :] = image_resized

        mask_padded = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
        mask_padded[:new_h, :new_w] = mask_resized

        # 6. 【核心修改】将bbox作为prompt（保留point变量名，兼容接口）
        point_prompt = bbox_resized.astype(np.float32)

        # 7. 数据增强（仅做不改变几何关系的增强）
        if np.random.random() > 0.5:
            # 水平翻转
            image_padded = np.fliplr(image_padded).copy()
            mask_padded = np.fliplr(mask_padded).copy()
            # 翻转bbox的x坐标
            point_prompt[0] = self.input_size - point_prompt[0]
            point_prompt[2] = self.input_size - point_prompt[2]

        # 8. 转换为Tensor并归一化（SAM官方参数）
        image_tensor = torch.from_numpy(image_padded).permute(2, 0, 1).float() / 255.0
        # 应用ImageNet均值标准差
        for t, m, s in zip(image_tensor, Config.MEAN, Config.STD):
            t.sub_(m).div_(s)

        mask_tensor = torch.from_numpy(mask_padded).float() / 255.0
        point_tensor = torch.from_numpy(point_prompt).float()  # 实际存储bbox

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "point": point_tensor,  # 变量名不变，兼容接口
            "orig_size": (orig_h, orig_w)
        }


class DiceLoss(torch.nn.Module):
    """Dice Loss（针对茎秆+雄穗优化）"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)


class SamTrainingManagerV2:
    def __init__(self, sam_manager=None):
        self.device = Config.DEVICE
        self.sam_manager = sam_manager
        self.model = None
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0

    def _load_sam_model(self):
        """加载SAM预训练模型"""
        if self.sam_manager and self.sam_manager.has_model_loaded():
            print("使用用户选择的SAM模型")
            self.model = self.sam_manager.model
            self.model.to(self.device)
        else:
            print(f"加载默认SAM模型: {Config.SAM_MODEL_TYPE}")
            self.model = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT)
            self.model.to(self.device)

        # 【核心优化】冻结策略：冻结image encoder，但解冻最后2层
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        # 解冻最后2层Transformer Block，让模型适配茎秆/雄穗的边缘特征
        for param in self.model.image_encoder.blocks[-2:].parameters():
            param.requires_grad = True
        print("SAM模型加载完成，已冻结image encoder前N层，解冻最后2层")

    def _compute_iou(self, pred_mask, true_mask):
        """计算前景IoU（仅针对茎秆+雄穗区域）"""
        pred = (pred_mask > 0.5).float()
        true = (true_mask > 0.5).float()
        intersection = (pred * true).sum()
        union = pred.sum() + true.sum() - intersection
        if union == 0:
            return torch.tensor(1.0, device=self.device)
        return intersection / union

    def _compute_loss(self, batch):
        """计算茎秆+雄穗专用损失（框选Prompt版）"""
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        points = batch["point"].to(self.device)  # 实际是bbox

        batch_size = images.shape[0]
        total_loss = 0.0
        total_iou = 0.0

        # 1. 一次性编码所有图像
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(images)

        # 2. 逐个样本处理
        for i in range(batch_size):
            # 【核心修改】使用bbox作为prompt
            bbox = points[i].unsqueeze(0)  # [1, 4] (x1,y1,x2,y2)

            # 编码提示（使用boxes替代points）
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None
            )

            # 4. 解码生成掩码
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings[i:i + 1],
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            # 5. 上采样到输入尺寸
            pred_masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=(Config.INPUT_SIZE, Config.INPUT_SIZE),
                mode="bilinear",
                align_corners=False
            )

            # 6. 【核心优化】计算损失（Dice为主，暴力前景权重）
            mask_gt = masks[i:i + 1].unsqueeze(1)

            # BCE Loss with 极端pos_weight
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_masks, mask_gt,
                pos_weight=torch.tensor([Config.POS_WEIGHT], device=self.device)
            )

            # Dice Loss
            dice_loss = self.dice_loss(pred_masks, mask_gt)

            # 组合损失：0.9 Dice + 0.1 BCE
            loss = 0.9 * dice_loss + 0.1 * bce_loss
            total_loss += loss

            # 计算前景IoU
            with torch.no_grad():
                iou = self._compute_iou(torch.sigmoid(pred_masks[0, 0]), mask_gt[0, 0])
                total_iou += iou

        return total_loss / batch_size, total_iou / batch_size

    def train(self, coco_container, image_paths, output_dir="sam_stem_tassel_models"):
        """开始训练（茎秆+雄穗专用）"""
        self._load_sam_model()
        os.makedirs(output_dir, exist_ok=True)

        # 初始化损失函数
        self.dice_loss = DiceLoss()

        # 准备数据
        full_dataset = SingleStemTasselDataset(coco_container, image_paths)
        if len(full_dataset) == 0:
            raise ValueError("没有有效的茎秆+雄穗训练数据，请检查标注是否完成")

        print(f"总训练样本数（单株茎秆+雄穗）: {len(full_dataset)}")

        # 【核心优化】按图片级拆分验证集
        # 先获取所有唯一图片路径
        unique_image_paths = list(set([s["image_path"] for s in full_dataset.samples]))
        np.random.shuffle(unique_image_paths)

        # 8:2拆分
        val_image_count = max(1, len(unique_image_paths) // 5)
        val_image_paths = set(unique_image_paths[:val_image_count])
        train_image_paths = set(unique_image_paths[val_image_count:])

        # 拆分数据集
        train_samples = [s for s in full_dataset.samples if s["image_path"] in train_image_paths]
        val_samples = [s for s in full_dataset.samples if s["image_path"] in val_image_paths]

        # 直接替换dataset的samples（简单高效）
        full_dataset.samples = train_samples
        train_dataset = full_dataset

        val_dataset = SingleStemTasselDataset(coco_container, image_paths)
        val_dataset.samples = val_samples

        print(f"训练集图片数: {len(train_image_paths)}, 单实例数: {len(train_dataset)}")
        print(f"验证集图片数: {len(val_image_paths)}, 单实例数: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS
        )

        # 优化器与学习率调度（茎秆+雄穗专用）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=Config.LEARNING_RATE, weight_decay=5e-6)
        scheduler = StepLR(optimizer, step_size=8, gamma=0.5)  # 每8轮学习率减半

        print(f"开始训练，Epochs: {Config.EPOCHS}, Device: {self.device}")

        for epoch in range(Config.EPOCHS):
            # 训练阶段
            self.model.train()
            train_loss_sum = 0.0
            train_iou_sum = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Train]")
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(pbar):
                loss, iou = self._compute_loss(batch)

                # 【核心优化】梯度累积
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                loss.backward()

                if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss_sum += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
                train_iou_sum += iou.item()
                pbar.set_postfix(
                    {"Loss": f"{loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}", "IoU": f"{iou.item():.4f}"})

            avg_train_loss = train_loss_sum / len(train_loader)
            avg_train_iou = train_iou_sum / len(train_loader)

            # 验证阶段
            self.model.eval()
            val_loss_sum = 0.0
            val_iou_sum = 0.0

            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Val]")
                for batch in pbar:
                    loss, iou = self._compute_loss(batch)
                    val_loss_sum += loss.item()
                    val_iou_sum += iou.item()
                    pbar.set_postfix({"Val Loss": f"{loss.item():.4f}", "Val IoU": f"{iou.item():.4f}"})

            avg_val_loss = val_loss_sum / len(val_loader)
            avg_val_iou = val_iou_sum / len(val_loader)

            # 更新学习率
            scheduler.step()

            # 打印日志
            print(f"\nEpoch {epoch + 1}/{Config.EPOCHS} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}, Val IoU:   {avg_val_iou:.4f}")
            print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型（基于Val IoU，比损失更可信）
            if avg_val_iou > self.best_val_iou:
                self.best_val_iou = avg_val_iou
                self.best_val_loss = avg_val_loss
                best_model_path = os.path.join(output_dir, "sam_stem_tassel_best.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  ✅ 保存最佳模型到: {best_model_path} (Val IoU: {avg_val_iou:.4f})")

            # 保存最新模型
            latest_model_path = os.path.join(output_dir, "sam_stem_tassel_latest.pth")
            torch.save(self.model.state_dict(), latest_model_path)

        print("\n🎉 训练完成！")
        print(f"最佳验证IoU: {self.best_val_iou:.4f}, 最佳验证损失: {self.best_val_loss:.4f}")

        # 使用最佳模型对验证集进行预测
        self._predict_val_set(val_loader, os.path.join(output_dir, "sam_stem_tassel_best.pth"))

        return os.path.join(output_dir, "sam_stem_tassel_best.pth")

    def _predict_val_set(self, val_loader, best_model_path):
        """使用最佳模型对验证集进行预测并保存结果（IndexError修复版）"""
        print("\n📊 开始对验证集进行预测...")

        # 加载最佳模型
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        # 创建输出目录
        output_dir = "sam_stem_tassel_val_img"
        os.makedirs(output_dir, exist_ok=True)

        # 对验证集进行预测
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                points = batch["point"].to(self.device)  # 实际是bbox
                orig_sizes = batch["orig_size"]

                batch_size = images.shape[0]

                # 一次性编码所有图像
                image_embeddings = self.model.image_encoder(images)

                for i in range(batch_size):
                    # 【核心修改】使用bbox作为prompt
                    bbox = points[i].unsqueeze(0)  # [1,4]

                    # 编码提示（使用boxes替代points）
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=bbox,
                        masks=None
                    )

                    # 解码生成掩码
                    low_res_masks, _ = self.model.mask_decoder(
                        image_embeddings=image_embeddings[i:i + 1],
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )

                    # 上采样到输入尺寸
                    pred_masks = torch.nn.functional.interpolate(
                        low_res_masks,
                        size=(Config.INPUT_SIZE, Config.INPUT_SIZE),
                        mode="bilinear",
                        align_corners=False
                    )

                    # 转换为二值掩码
                    pred_mask = (torch.sigmoid(pred_masks[0, 0]) > 0.5).cpu().numpy()

                    # 获取真值掩码
                    true_mask = masks[i].cpu().numpy()

                    # 获取原始图像（反归一化）
                    image = batch["image"][i].cpu().numpy()
                    # 反归一化（变量名i改为j，避免覆盖）
                    for j, (m, s) in enumerate(zip(Config.MEAN, Config.STD)):
                        image[j] = image[j] * s + m
                    image = (image * 255).astype(np.uint8).transpose(1, 2, 0)

                    # 调整大小到原始尺寸（修复IndexError核心逻辑）
                    try:
                        # 打印调试信息
                        print(f"orig_sizes type: {type(orig_sizes)}")
                        print(f"orig_sizes value: {orig_sizes}")
                        
                        # 兼容DataLoader collate后的不同格式
                        if isinstance(orig_sizes, torch.Tensor):
                            # 格式: [batch, 2] (h, w)
                            if orig_sizes.shape[0] > i:
                                size = orig_sizes[i].cpu().numpy()
                                if len(size) == 2:
                                    orig_h, orig_w = size
                                else:
                                    orig_h, orig_w = 512, 512
                            else:
                                orig_h, orig_w = 512, 512
                        elif isinstance(orig_sizes, tuple) and len(orig_sizes) == 2:
                            # 格式: ([h1, h2...], [w1, w2...])
                            if len(orig_sizes[0]) > i and len(orig_sizes[1]) > i:
                                orig_h = orig_sizes[0][i].item() if hasattr(orig_sizes[0][i], 'item') else orig_sizes[0][i]
                                orig_w = orig_sizes[1][i].item() if hasattr(orig_sizes[1][i], 'item') else orig_sizes[1][i]
                            else:
                                orig_h, orig_w = 512, 512
                        elif isinstance(orig_sizes, list):
                            # 格式: [(h1,w1), (h2,w2)...] 或 [h1,h2...]/[w1,w2...]
                            if i < len(orig_sizes):
                                item = orig_sizes[i]
                                if isinstance(item, (list, tuple)) and len(item) == 2:
                                    orig_h, orig_w = item
                                else:
                                    # 单个值的情况，使用默认值
                                    orig_h, orig_w = 512, 512
                            else:
                                orig_h, orig_w = 512, 512
                        else:
                            # 其他格式，使用默认值
                            orig_h, orig_w = 512, 512
                    except Exception as e:
                        # 所有异常兜底
                        print(f"Error getting original size: {e}")
                        orig_h, orig_w = 512, 512

                    image = cv2.resize(image, (orig_w, orig_h))
                    pred_mask = cv2.resize(pred_mask.astype(np.float32), (orig_w, orig_h)) > 0.5
                    true_mask = cv2.resize(true_mask.astype(np.float32), (orig_w, orig_h)) > 0.5

                    # 保存结果
                    base_filename = f"val_{batch_idx}_{i}"

                    # 保存原图
                    orig_path = os.path.join(output_dir, f"{base_filename}_orig.jpg")
                    Image.fromarray(image).save(orig_path, quality=85, optimize=True)

                    # 保存预测掩码
                    pred_path = os.path.join(output_dir, f"{base_filename}_pred.png")
                    pred_mask_img = (pred_mask * 255).astype(np.uint8)
                    Image.fromarray(pred_mask_img).save(pred_path, optimize=True, compress_level=9)

                    # 保存真值掩码
                    true_path = os.path.join(output_dir, f"{base_filename}_true.png")
                    true_mask_img = (true_mask * 255).astype(np.uint8)
                    Image.fromarray(true_mask_img).save(true_path, optimize=True, compress_level=9)

                    # 保存叠加效果（绿色=预测，红色=真值）
                    overlay_path = os.path.join(output_dir, f"{base_filename}_overlay.jpg")
                    overlay = image.copy()
                    overlay[pred_mask] = (0, 255, 0)  # 绿色预测
                    overlay[true_mask] = (255, 0, 0)  # 红色真值
                    Image.fromarray(overlay).save(overlay_path, quality=85, optimize=True)

                    print(f"保存验证集预测结果: {base_filename}")

        print(f"\n✅ 验证集预测完成，结果已保存到: {output_dir}")

    def start_training(self, coco_container, image_paths):
        """开始训练（兼容旧接口）"""
        try:
            best_model_path = self.train(coco_container, image_paths)
            return True, f"训练完成，最佳模型已保存到: {best_model_path}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

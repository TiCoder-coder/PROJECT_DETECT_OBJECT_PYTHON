# ================================================= TRAIN CO TICH HOP SAM2 && YOLOV11 =================================================

import os, sys
import json
import ijson
import torch.nn.functional as F
import random
import cv2
import math
from datetime import datetime
import time
import gc
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from omegaconf import OmegaConf
from typing import Dict, Optional, List, Any

import os, sys
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from sam2.sam2.modeling.sam2_base import SAM2Base
    from sam2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
    from sam2.sam2.build_sam import build_sam2
    from sam2.training.trainer import Trainer
    from sam2.sam2.sam2_train import SAM2Train

    print("[SUCCESS] SAM2 imported OK.")
except Exception as e:
    print("[ERROR] Import failed:", e)
    raise

import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO


Y_NEW = os.path.join(PROJECT_ROOT, "runs", "train", "yolo11_finetune_hybrid", "weights", "best.pt")
Y_FALLBACK = os.path.join(PROJECT_ROOT, "checkpoints", "yolo11n.pt")
CHECKPOINT_YOLO = Y_NEW if os.path.exists(Y_NEW) else Y_FALLBACK
if CHECKPOINT_YOLO == Y_FALLBACK:
    print("[INFO] YOLO fine-tune is not available yet, use the original yolo11n.pt.")
    print("→ Please run the following command if you want to tweak YOLO:")
    print("   yolo detect train model=checkpoints/yolo11n.pt "
          "data=data/hybrid_data_yolo/hybrid_data_yolo.yaml "
          "epochs=50 imgsz=640 batch=16 device=0 project=runs/train name=yolo11_finetune_hybrid\n")
else:
    print(f"[INFO] Use YOLO fine-tune: {CHECKPOINT_YOLO}")

print(f"[INFO] Loading YOLOv11 model from: {CHECKPOINT_YOLO}")
yolo_guidance = YOLO(CHECKPOINT_YOLO)
yolo_guidance.to("cuda" if torch.cuda.is_available() else "cpu")
yolo_guidance.fuse()
yolo_guidance.model.float()

torch.multiprocessing.set_sharing_strategy('file_system')

class CustomDataset(Dataset):
    def __init__(self, img_dir, coco_json_path, augment=True, use_cache=False,
                 chunk_index=0, chunk_size=999999, device=None):
        self.img_dir = img_dir
        self.coco_json_path = coco_json_path
        self.augment = augment
        self.use_cache = use_cache
        self.chunk_index = chunk_index
        self.chunk_size = chunk_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        self.images = coco.get("images", [])
        
        valid_images = []
        for img in self.images:
            img_name = img["file_name"]
            img_path = os.path.join(self.img_dir, img_name)
            
            if os.path.exists(img_path):
                valid_images.append(img)
            else:
                dir_name, file_base_name = os.path.split(img_name)
                base_name_no_ext, ext = os.path.splitext(file_base_name)
                
                viz_file_name = os.path.join(dir_name, base_name_no_ext + "_viz" + ext)
                viz_img_path = os.path.join(self.img_dir, viz_file_name)

                if os.path.exists(viz_img_path):
                    img["file_name"] = viz_file_name
                    valid_images.append(img)
                    print(f"[AUTO-FIX] Find images with the suffix '_viz' and use: {viz_img_path}")
                else:
                    print(f"[CLEAN] Remove error or non-existent images: {img_path} (Và {viz_img_path})")
                    
        self.images = valid_images
                
        if len(self.images) == 0:
            print(f"[AUTO-FIX] No valid photo in {self.coco_json_path} → try using image from folder {self.img_dir}")
            extra_imgs = []
            for root, _, files in os.walk(self.img_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        extra_imgs.append({
                            "id": len(extra_imgs) + 1,
                            "file_name": os.path.relpath(os.path.join(root, f), self.img_dir)
                        })
            if extra_imgs:
                self.images = extra_imgs
                print(f"[AUTO-FIX] Restore {len(self.images)} photo from folder {self.img_dir}.")

        print(f"[DATASET] {len(self.images)} valid image after filtering.")

        self.annotations = coco.get("annotations", [])
        self.categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

        self.id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.id_to_anns:
                self.id_to_anns[img_id] = []
            self.id_to_anns[img_id].append(ann)

        if augment:
            self.transform = A.Compose([
                A.Resize(640, 640),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomRotate90(p=0.2),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(640, 640),
                ToTensorV2(),
            ])

        print(f"[DATASET] Loaded {len(self.images)} images from {os.path.basename(coco_json_path)}")
        print(f"[DATASET] Found {len(self.categories)} categories: {list(self.categories.values())}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_name = img_info["file_name"]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            base_name, ext = os.path.splitext(img_name)
            candidate_suffixes = ["", "_viz", "_mask", "_seg"]

            found = False
            for suffix in candidate_suffixes:
                for candidate_ext in [".jpg", ".jpeg", ".png"]:
                    candidate_path = os.path.join(self.img_dir, f"{base_name}{suffix}{candidate_ext}")
                    if os.path.exists(candidate_path):
                        img_path = candidate_path
                        print(f"[AUTO-FIX] Use alternative images with suffixes: {os.path.basename(candidate_path)}")
                        found = True
                        break
                if found:
                    break

            if not found:
                search = glob.glob(os.path.join(self.img_dir, "**", f"{base_name}*.*"), recursive=True)
                if search:
                    img_path = search[0]
                    print(f"[AUTO-SEARCH] Similar files found: {os.path.basename(img_path)}")
                else:
                    print(f"[WARN] No suitable images found for {img_name}")
                    return None


        image = cv2.imread(img_path)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARN] Can't read the image: {img_path}")

            cls_name = os.path.basename(os.path.dirname(img_path))
            cls_dir = os.path.join(self.img_dir, cls_name)

            candidate_imgs = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and os.path.exists(os.path.join(cls_dir, f))
                and f != os.path.basename(img_path)
            ]

            if candidate_imgs:
                replacement = os.path.join(cls_dir, random.choice(candidate_imgs))
                print(f"[RECOVER] Use another photo instead: {replacement}")
                image = cv2.imread(replacement)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    print(f"[FAIL] The replacement photo is also faulty.: {replacement}")
                    return None
            else:
                print(f"[SKIP] No replacement image found in {cls_dir}")
                return None
            backup_imgs = [
                os.path.join(self.img_dir, i["file_name"])
                for i in self.images
                if i["file_name"] != img_name and os.path.exists(os.path.join(self.img_dir, i["file_name"]))
            ]
            if backup_imgs:
                replacement = random.choice(backup_imgs)
                print(f"[RECOVER-JSON] Replace with another image in COCO JSON: {replacement}")
                image = cv2.imread(replacement)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    print(f"[FAIL] The replacement image from JSON also fails: {replacement}")
                    return None
            else:
                print("[SKIP] There are no valid images in the entire dataset..")
                return None

        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        anns = self.id_to_anns.get(img_info["id"], [])
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)

        aug = self.transform(image=image, mask=mask)
        img = aug["image"].float() / 255.0
        mask = aug["mask"].unsqueeze(0).float()

        return {
            "image": img,
            "mask": mask,
            "filename": img_name,
            "original_size": (h, w),
        }


def filtered_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        print("[WARN] Empty patch → dummy")
        dummy_img = torch.zeros((1, 1, 3, 1024, 1024), dtype=torch.float32)
        dummy_mask = torch.zeros((1, 1, 1, 1024, 1024), dtype=torch.bool)
        dummy_obj_to_frame_idx = torch.tensor([[[0, 0]]], dtype=torch.int)
        dummy_meta = BatchedVideoMetaData(
            unique_objects_identifier=torch.zeros((1, 1, 3), dtype=torch.long),
            frame_orig_size=torch.zeros((1, 1, 2), dtype=torch.long),
        )
        masks = masks.squeeze(0)
        return BatchedVideoDatapoint(
            img_batch=dummy_img,
            masks=dummy_mask,
            obj_to_frame_idx=dummy_obj_to_frame_idx,
            metadata=dummy_meta,
            dict_key="image",
            batch_size=[1],
        )

    images = []
    masks = []
    for b in batch:
        img = b['image']
        mask = b['mask']
        if img.ndim == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        images.append(img.float())

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        masks.append(mask.float())

    images = torch.stack(images)
    masks = torch.stack(masks)

    images = images.unsqueeze(0)
    masks = masks.unsqueeze(0)
    masks = masks.squeeze(0)

    T = images.shape[0]
    B = images.shape[1]

    obj_to_frame_idx = torch.tensor([[[0, i] for i in range(B)]], dtype=torch.int)

    unique_objects_identifier = torch.zeros((T, B, 3), dtype=torch.long)
    frame_orig_size = torch.tensor(
        [[[images.shape[3], images.shape[4]]] * B], dtype=torch.long
    )

    metadata = BatchedVideoMetaData(
        unique_objects_identifier=unique_objects_identifier,
        frame_orig_size=frame_orig_size,
    )

    batch_dp = BatchedVideoDatapoint(
        img_batch=images,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks.bool(),
        metadata=metadata,
        dict_key="image",
        batch_size=[T],
    )

    print(f"[DEBUG] Collated OK → img_batch shape: {batch_dp.img_batch.shape}")
    print(f"[DEBUG] flat_img_batch shape: {batch_dp.flat_img_batch.shape}")
    return batch_dp


def simple_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    filenames = [item["filename"] for item in batch]
    sizes = [item["original_size"] for item in batch]

    return {
        "image": images,
        "mask": masks,
        "filename": filenames,
        "original_size": sizes,
    }

class OptimizedTrainer:
    def __init__(self, config_path: str, output_path: str, device: Optional[str] = None):
        OmegaConf.register_new_resolver("divide", lambda a, b: float(a) / float(b))
        OmegaConf.register_new_resolver("times", lambda *args: float(np.prod([float(x) for x in args])))
        OmegaConf.register_new_resolver("add", lambda a, b: float(a) + float(b))
        OmegaConf.register_new_resolver("sub", lambda a, b: float(a) - float(b))
        OmegaConf.register_new_resolver("int", lambda x: int(x))
        OmegaConf.register_new_resolver("ceil_int", lambda x: int(math.ceil(float(x))))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_path = output_path
        self.checkpoint_dir = os.path.join(output_path, "experiments", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        cfg = OmegaConf.load(config_path)
        if "trainer" not in cfg:
            cfg = OmegaConf.create({
                "trainer": {"model": {}, "data": {}, "max_epochs": 10}
            })

        trainer_cfg = cfg.get("trainer", OmegaConf.create({
            "model": cfg.get("model", cfg),
            "data": {"train": {}, "val": {}},
            "logging": {"level": "info"},
            "checkpoint": {"save_dir": self.checkpoint_dir},
            "max_epochs": 10,
            "optim": {}
        }))


        try:
            self.output_path = output_path
            self.config_path = config_path
            

            cfg = OmegaConf.load(config_path)
            trainer_cfg = cfg.trainer
            
            self.device = torch.device(trainer_cfg.accelerator)
            
            SAM2_ARCH_CONFIG_PATH = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/configs/sam2.1/sam2.1_hiera_b+.yaml"
            
            model_path = None
            
            print(f"[INFO] Building SAM2 model from architecture config: {SAM2_ARCH_CONFIG_PATH}")
            base_model = build_sam2(SAM2_ARCH_CONFIG_PATH, model_path, device=self.device)

            try:
                self.model = SAM2Train(base_model)
                print("[INFO] SAM2Train wrapper applied successfully.")
            except Exception as e:
                print("[ERROR] Failed to wrap model with SAM2Train:", e)
                raise

            self.trainer = Trainer(
                model=self.model,
                data=trainer_cfg.data,
                logging=trainer_cfg.logging,
                checkpoint=trainer_cfg.checkpoint,
                max_epochs=trainer_cfg.max_epochs,
                optim=trainer_cfg.optim,
                mode=trainer_cfg.get("mode", "train_only"),
                accelerator=trainer_cfg.get("accelerator", "cuda"),
                seed_value=trainer_cfg.get("seed_value", 123),
                val_epoch_freq=trainer_cfg.get("val_epoch_freq", 1),
            )


            if hasattr(self.trainer.model, "gradient_checkpointing_enable"):
                self.trainer.model.gradient_checkpointing_enable()
            print("[INFO] Gradient checkpointing enabled.")

        except Exception as e:
            print("[ERROR] SAM2Train init failed:", e)
            raise


        self.scaler = GradScaler(
            'cuda',
            enabled=self.device.type == 'cuda',
            init_scale=2.**16,
            growth_interval=2000
        )
        print(f"[INFO] Trainer initialized on {self.device}")



CHECKPOINT_EVERY_STEPS = 2
CHECKPOINT_MAX_SIZE_GB = 6.0

def find_latest_run_index(checkpoint_dir):
    run_dirs = glob.glob(os.path.join(checkpoint_dir, "run_*"))
    if not run_dirs:
        return 0
    
    max_index = 0
    for r_dir in run_dirs:
        try:
            index_str = os.path.basename(r_dir).split('_')[-1]
            index = int(index_str)
            if index > max_index:
                max_index = index
        except:
            continue
    return max_index

def get_current_checkpoint_file(run_dir):
    if not os.path.isdir(run_dir):
        return None
        
    files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]
    if not files:
        return None
        
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(run_dir, x)))
    path = os.path.join(run_dir, latest)
    
    if os.path.getsize(path) / (1024**3) > CHECKPOINT_MAX_SIZE_GB:
        return None
    return path

class DummyDataset:
    def __init__(self, loader):
        self.loader = loader
    def get_loader(self, epoch):
        return self.loader
# ====================== MAIN ======================
if __name__ == "__main__":
    img_dir = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/hybrid_data"
    ann_dir = img_dir
    output_path = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/output"
    config_path = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/configs/yolo/yolo_learning_tools.yaml"

    print("[WARN] If the dataset is very large, make sure the hybrid_coco.json file is split or reduce augmentations to avoid OOM.")

    trainer = OptimizedTrainer(config_path=config_path, output_path=output_path)
    print("[INFO] Freezing SAM2 backbone layers for lighter training...")

    for name, param in trainer.trainer.model.named_parameters():
        param.requires_grad = any(k in name for k in ["decoder", "mask_head"])

    if hasattr(trainer.trainer.model, "gradient_checkpointing_enable"):
        trainer.trainer.model.gradient_checkpointing_enable()
    print("[INFO] SAM2 backbone frozen (only decoder and mask head will train).")

    latest_run_idx = find_latest_run_index(trainer.checkpoint_dir)
    current_run_idx = latest_run_idx + 1
    run_dir = os.path.join(trainer.checkpoint_dir, f"run_{current_run_idx:03d}")
    os.makedirs(run_dir, exist_ok=True)

    latest_ckpt_path = None
    initial_epoch = 1

    if latest_run_idx > 0:
        prev_run_dir = os.path.join(trainer.checkpoint_dir, f"run_{latest_run_idx:03d}")
        latest_ckpt_path = get_current_checkpoint_file(prev_run_dir)
        if latest_ckpt_path:
            print(f"[RESUME] Loading checkpoint from run {latest_run_idx}: {latest_ckpt_path}")
            checkpoint = torch.load(latest_ckpt_path, map_location=trainer.device)
            trainer.trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            initial_epoch = checkpoint.get("epoch", 0) + 1
            print(f"[RESUME] Done. New run {current_run_idx:03d}, start epoch {initial_epoch}.")
        else:
            print(f"[INFO] No old checkpoint, start new Run {current_run_idx:03d}.")
    else:
        print(f"[INFO] First train, start Run {current_run_idx:03d}.")

    json_file = os.path.join(ann_dir, "hybrid_coco.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"[ERROR] File not found {json_file}")

    subset_ds = CustomDataset(img_dir=img_dir, coco_json_path=json_file, augment=True)
    sorted_items = sorted(subset_ds.categories.items(), key=lambda x: int(x[0]))
    class_names = [name for (_id, name) in sorted_items]
    class_to_idx = {name: int(_id) for (_id, name) in sorted_items}
    train_config = OmegaConf.to_container(OmegaConf.load(config_path))

    train_loader = DataLoader(
        subset_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, collate_fn=simple_collate
    )

    print(f"[DATASET] Total number of valid photos: {len(subset_ds)}")

    model = trainer.trainer.model
    device = trainer.device
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    if latest_ckpt_path and initial_epoch > 1:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print("[RESUME] Loaded optimizer và scaler states.")

    scaler = trainer.scaler
    num_epochs = initial_epoch + 12

    best_loss = float('inf')
    best_state = None

    print(f"[TRAIN] Start training from epoch {initial_epoch} → {num_epochs}...")

    for epoch in range(initial_epoch, num_epochs + 1):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            if batch is None:
                continue

            imgs = batch["image"].to(device)
            gt = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    yolo_preds = yolo_guidance(imgs)
                yolo_masks = torch.zeros((imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]), device=device)
                for i, pred in enumerate(yolo_preds):
                    for j, box in enumerate(pred.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        if j >= yolo_masks.shape[1]:
                            break
                        yolo_masks[i, j, y1:y2, x1:x2] = 1.0

                B, _, H, W = imgs.shape
                meta = BatchedVideoMetaData(
                    unique_objects_identifier=torch.zeros((1, B, 3), dtype=torch.long, device=device),
                    frame_orig_size=torch.tensor([[[H, W]] * B], dtype=torch.long, device=device),
                )
                obj_to_frame_idx = torch.tensor([[[0, i] for i in range(B)]], dtype=torch.int, device=device)
                batched_input = BatchedVideoDatapoint(
                    img_batch=imgs.unsqueeze(0),
                    obj_to_frame_idx=obj_to_frame_idx,
                    masks=yolo_masks.unsqueeze(0),
                    metadata=meta,
                    dict_key="image",
                    batch_size=[B],
                )
                out = model.forward_train(batched_input) if hasattr(model, "forward_train") else model.forward(batched_input)

            pred = None
            if isinstance(out, dict):
                pred = out.get("pred_masks", out.get("masks", out.get("low_res_masks", None)))

            if pred is None:
                raise ValueError(f"[ERROR] Không lấy được pred_masks từ model output.")

            if pred.dim() == 4 and pred.shape[1] > 1:
                pred = pred[:, 0:1, :, :]
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            if (pred.shape[-2], pred.shape[-1]) != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)

            gt = gt.float()
            loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f"[Epoch {epoch:02d} | Step {step:04d}] loss={loss.item():.5f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"[Epoch {epoch:02d}] → avg_loss={avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": avg_loss,
                "classes": class_names,
                "class_to_idx": class_to_idx,
                "train_config": train_config,
            }
            print(f"New best model saved (loss={best_loss:.5f})")

    final_ckpt = os.path.join(output_path, "sam2_finetuned_final.pth")

    if best_state is None:
        print("\nKhông có epoch nào tạo ra best_state hợp lệ — có thể dataset rỗng hoặc training bị bỏ qua.")
        best_state = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": float('inf'),
            "classes": class_names,
            "class_to_idx": class_to_idx,
            "train_config": train_config,
        }

    torch.save(best_state, final_ckpt, _use_new_zipfile_serialization=False)
    print("\nTraining completed!")
    print("The best model is saved at:", final_ckpt)


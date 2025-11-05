import os, sys, json, cv2, torch, yaml, gc
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from sam2.sam2.modeling.sam2_base import SAM2Base
    from sam2.training.trainer import Trainer as SAMTrainer
    from sam2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
    from automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2.build_sam import build_sam2
    print("[SUCCESS] SAM2 imported OK.")
except Exception as e:
    print("[ERROR] Import failed:", e)
    raise

CHECKPOINT_YOLO = os.path.join(PROJECT_ROOT, "checkpoints", "yolo11n.pt")
CHECKPOINT_SAM = os.path.join(PROJECT_ROOT, "checkpoints", "sam2.1_hiera_base_plus.pt")
CONFIG_SAM = os.path.join(PROJECT_ROOT, "configs", "sam2.1", "sam2.1_hiera_b+.yaml")

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

HYBRID_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "hybrid_data")
YOLO_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "hybrid_data_yolo")
os.makedirs(HYBRID_DATA_DIR, exist_ok=True)
os.makedirs(YOLO_DATA_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] Using CPU (annotation will be slow)")

print(f"[INFO] Loading YOLOv11 from: {CHECKPOINT_YOLO}")
yolo = YOLO(CHECKPOINT_YOLO)

print(f"[INFO] Loading SAM2.1 from: {CHECKPOINT_SAM}")
sam_model = build_sam2(CONFIG_SAM, CHECKPOINT_SAM, device=DEVICE)
mask_generator = SAM2AutomaticMaskGenerator(
    sam_model,
    points_per_side=32,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.92,
    min_mask_region_area=100,
)
print("[READY] Both YOLOv11 and SAM2.1 loaded successfully.")

images, annotations, categories = [], [], {}
image_id, ann_id = 1, 1

print(f"[START] Scanning folder: {RAW_DIR}")

images, annotations, categories = [], [], {}
image_id, ann_id = 1, 1

print(f"[START] Scanning folder: {RAW_DIR}")

def nms_numpy(boxes, iou_thresh=0.5):

    if len(boxes) == 0:
        return boxes
    boxes = boxes[np.argsort(boxes[:, 3])]
    keep = []
    while len(boxes) > 0:
        box = boxes[-1]
        keep.append(box)
        boxes = boxes[:-1]
        if len(boxes) == 0:
            break
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        iou = inter_area / (union_area + 1e-6)
        boxes = boxes[iou < iou_thresh]
    return np.array(keep)


for root, _, files in os.walk(RAW_DIR):
    label = os.path.basename(root)
    if label == os.path.basename(RAW_DIR):
        continue

for root, _, files in os.walk(RAW_DIR):
    label = os.path.basename(root)
    if label == os.path.basename(RAW_DIR):
        continue

    if label not in categories:
        categories[label] = len(categories) + 1

    for file_name in tqdm(files, desc=f"Processing {label}"):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path = os.path.join(root, file_name)
        img = cv2.imread(file_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        results = yolo(img, conf=0.4, device=DEVICE)
        raw_boxes = results[0].boxes
        if not len(raw_boxes):
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            continue

        has_valid_box = False
        overlay = img.copy()

        boxes = raw_boxes.xyxy.cpu().numpy()
        confidences = raw_boxes.conf.cpu().numpy()

        boxes = nms_numpy(boxes, iou_thresh=0.5)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(confidences[i]) if i < len(confidences) else 0.0
            if conf < 0.4:
                continue

            pad = 15
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            try:
                with torch.no_grad():
                    masks = mask_generator.generate(crop)
            except torch.OutOfMemoryError:
                print(f"[OOM] Skip box {x1},{y1},{x2},{y2}")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue

            if not masks:
                continue

            def mask_iou(mask, box):
                x1, y1, x2, y2 = box
                mask_box = np.zeros(mask.shape[:2], dtype=np.uint8)
                mask_box[y1:y2, x1:x2] = 1
                inter = np.logical_and(mask > 0, mask_box > 0).sum()
                union = np.logical_or(mask > 0, mask_box > 0).sum()
                return inter / (union + 1e-6)

            best_mask = max(masks, key=lambda m: mask_iou(np.array(m["segmentation"], dtype=np.uint8),
                                                        [0, 0, crop.shape[1], crop.shape[0]]))
            segmentation = np.array(best_mask["segmentation"], dtype=np.uint8)

            seg_coords = np.column_stack(np.where(segmentation > 0))
            seg_coords[:, [0, 1]] = seg_coords[:, [1, 0]]
            seg_coords[:, 0] += x1p
            seg_coords[:, 1] += y1p
            segmentation_flat = seg_coords.flatten().tolist()


            bbox = [x1, y1, x2 - x1, y2 - y1]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": categories[label],
                "segmentation": [segmentation_flat],
                "bbox": bbox,
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
                "confidence": conf
            })
            ann_id += 1
            has_valid_box = True

            color = (0, 255, 0)
            mask_rgb = np.zeros_like(img, dtype=np.uint8)

            seg_h, seg_w = segmentation.shape[:2]
            crop_h, crop_w = y2 - y1, x2 - x1
            if seg_h != crop_h or seg_w != crop_w:
                segmentation_vis = cv2.resize(segmentation, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            else:
                segmentation_vis = segmentation

            mask_region = (segmentation_vis > 0).astype(np.uint8)
            mask_rgb[y1:y2, x1:x2][mask_region == 1] = color

            overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{label} ({conf:.2f})", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{label} ({conf:.2f})", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if has_valid_box:
            label_dir = os.path.join(HYBRID_DATA_DIR, label)
            os.makedirs(label_dir, exist_ok=True)
            viz_name = f"{os.path.splitext(file_name)[0]}_viz.jpg"
            save_path = os.path.join(label_dir, viz_name)
            cv2.imwrite(save_path, overlay)
            rel_path = os.path.relpath(save_path, HYBRID_DATA_DIR)
            images.append({
                "id": image_id,
                "file_name": rel_path,
                "width": w,
                "height": h
            })
            image_id += 1
        gc.collect()


COCO_PATH = os.path.join(HYBRID_DATA_DIR, "hybrid_coco.json")
LABEL_PATH = os.path.join(HYBRID_DATA_DIR, "label.json")

coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": cid, "name": name} for name, cid in categories.items()]
}

with open(COCO_PATH, "w") as f:
    json.dump(coco_output, f, indent=4)
with open(LABEL_PATH, "w") as f:
    json.dump(categories, f, indent=4)

print(f"\n[SUCCESS] Hybrid annotation saved successfully!")
print(f" → COCO JSON: {COCO_PATH}")
print(f" → Label map: {LABEL_PATH}")
print(f" → Classes ({len(categories)}): {list(categories.keys())}")
print(f" → Total images: {len(images)} | Total annotations: {len(annotations)}")

print("Done — annotation process completed successfully.")
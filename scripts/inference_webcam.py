import os, sys, time, glob, cv2, json, numpy as np, torch
from collections import defaultdict, OrderedDict
import torchvision.transforms as T
from torchvision import models
import torch.nn.functional as F
from ultralytics import YOLO

BASE_DIR = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT"
PATHS = {
    "ANNOTATED_DIR": os.path.join(BASE_DIR, "data", "hybrid_data"),
    "SAM2_CKPT": os.path.join(BASE_DIR, "output", "sam2_finetuned_final.pth"),
    "SAM2_CONFIG": os.path.join(BASE_DIR, "configs", "sam2.1", "sam2.1_hiera_b+.yaml"),
}
for k, v in PATHS.items():
    print(f"[{'OK' if os.path.exists(v) else 'MISS'}] {k}: {v}")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True
print("[INFO] device:", device)

PROJECT_ROOT = BASE_DIR
SAM2_ROOT = os.path.join(PROJECT_ROOT, "sam2")
for p in [SAM2_ROOT, os.path.join(SAM2_ROOT, "sam2")]:
    if p not in sys.path: sys.path.insert(0, p)

from build_sam import build_sam2
from sam2_image_predictor import SAM2ImagePredictor

def load_sam2_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    info = {}
    if isinstance(ckpt, dict):
        info['classes'] = ckpt.get("classes", [])
        sd = ckpt.get("model_state_dict", ckpt)
    else:
        sd = ckpt
    new_sd = OrderedDict((k[7:], v) if k.startswith("module.") else (k, v) for k, v in sd.items())
    return new_sd, info

def build_backbone(pretrained=True):
    try:
        if hasattr(models, 'resnet18') and pretrained:
            backbone = models.resnet18(weights=getattr(models, "ResNet18_Weights", None).DEFAULT) if hasattr(models, "ResNet18_Weights") else models.resnet18(pretrained=True)
        else:
            backbone = models.resnet18(weights=None)
    except Exception:
        backbone = models.resnet18(weights=None)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-1])).to(device).eval()
    return backbone

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_prototypes(annotated_dir, classes_in_ckpt, backbone, max_per_class=100):
    prototypes = defaultdict(list)
    img_patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files = []
    for pat in img_patterns:
        files.extend(glob.glob(os.path.join(annotated_dir, pat), recursive=True))
    for p in files:
        name = os.path.basename(p).lower()
        parent = os.path.basename(os.path.dirname(p)).lower()
        matched = None
        for cls in classes_in_ckpt:
            cl = cls.lower()
            if cl in name or cl == parent or cl in parent or cl in name:
                matched = cls
                break
        if not matched:
            continue
        if len(prototypes[matched]) >= max_per_class:
            continue
        img = cv2.imread(p)
        if img is None: continue
        try:
            inp = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = backbone(inp).squeeze().cpu().numpy()
        except Exception as e:
            continue
        prototypes[matched].append(emb)
    proto_mean = {}
    for cls, embs in prototypes.items():
        if len(embs) == 0: continue
        proto_mean[cls] = np.mean(np.stack(embs, 0), axis=0)
    print(f"[INFO] Built prototypes for {len(proto_mean)} / {len(classes_in_ckpt)} classes")
    return proto_mean

def generate_grid_boxes(w, h, scales=(0.15, 0.3, 0.45, 0.6), stride=150):
    boxes = []
    for s in scales:
        bw = int(w * s); bh = int(h * s)
        if bw < 32: bw = 32
        if bh < 32: bh = 32
        step_x = max(int(bw * 0.6), stride)
        step_y = max(int(bh * 0.6), stride)
        for x in range(0, w - bw + 1, step_x):
            for y in range(0, h - bh + 1, step_y):
                boxes.append([x, y, x + bw, y + bh])
    return np.array(boxes)

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter + 1e-8
    return inter / union

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def nms_boxes(boxes, scores, iou_thresh=0.3):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]; keep.append(i)
        rest = idxs[1:]
        rem = []
        for j in rest:
            if box_iou(boxes[i], boxes[j]) > iou_thresh:
                continue
            rem.append(j)
        idxs = np.array(rem, dtype=int)
    return keep

class SAM2Detector:
    def __init__(self, ckpt_path, cfg_path):
        sd, info = load_sam2_weights(ckpt_path)
        self.classes = info.get("classes", [])
        print(f"[INFO] CKPT classes count: {len(self.classes)}")
        self.model = build_sam2(cfg_path, None, device)
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.float().to(device).eval()
        self.predictor = SAM2ImagePredictor(self.model)

        self.backbone = build_backbone(pretrained=True)
        self.prototypes = build_prototypes(PATHS["ANNOTATED_DIR"], self.classes, self.backbone, max_per_class=100)
        for k in list(self.prototypes.keys()):
            v = self.prototypes[k].ravel()
            self.prototypes[k] = v / (np.linalg.norm(v) + 1e-8)

    def _match_class(self, crop):
        inp = preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.backbone(inp).squeeze().cpu().numpy().ravel()
        qn = q / (np.linalg.norm(q) + 1e-8)
        best_cls = None; best_sc = -1.0
        for cls, proto in self.prototypes.items():
            sc = float(np.dot(qn, proto))
            if sc > best_sc:
                best_sc = sc; best_cls = cls
        return best_cls, best_sc, qn

    @torch.inference_mode()
    def infer_frame(self, frame,
                    box_sc_th=0.35,
                    mask_area_th=600,
                    proto_sim_th=0.6,
                    stride=200,
                    scales=(0.18, 0.3, 0.45)):
        """
        Phiên bản tối ưu realtime:
        - Tạo lưới box vừa phải, bỏ vùng trống (low variance)
        - Predict batch nhỏ với SAM2
        - Ghép nhiều vật thể riêng biệt
        """
        H, W = frame.shape[:2]
        rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale_factor = 0.6
        small_H, small_W = int(H * scale_factor), int(W * scale_factor)
        small_frame = cv2.resize(rgb_small, (small_W, small_H))

        grid_boxes = []
        for s in scales:
            bw, bh = int(small_W * s), int(small_H * s)
            for x in range(0, small_W - bw, stride):
                for y in range(0, small_H - bh, stride):
                    crop = small_frame[y:y + bh, x:x + bw]
                    if crop.size == 0:
                        continue
                    if np.var(crop) < 20:
                        continue
                    grid_boxes.append([x, y, x + bw, y + bh])
        if not grid_boxes:
            return []

        boxes = np.array(grid_boxes, dtype=np.int32)
        if len(boxes) > 60:
            idx = np.random.choice(len(boxes), 60, replace=False)
            boxes = boxes[idx]

        detections = []
        self.predictor.set_image(small_frame)

        batch_size = 10
        for i in range(0, len(boxes), batch_size):
            sub_boxes = boxes[i:i + batch_size]
            try:
                out = self.predictor.predict(box=sub_boxes, multimask_output=False)
                if isinstance(out, tuple):
                    masks, scores = out[:2]
                else:
                    continue
                masks = np.asarray(masks)
                scores = np.asarray(scores)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
            except Exception:
                continue

            for j, sc in enumerate(scores):
                if float(sc) < box_sc_th:
                    continue
                mask = masks[j].astype(np.uint8)
                area = int(mask.sum())
                if area < mask_area_th:
                    continue
                ys, xs = np.where(mask > 0)
                if ys.size == 0:
                    continue
                bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                bx1, by1, bx2, by2 = map(lambda z: int(z / scale_factor), [bx1, by1, bx2, by2])
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(W - 1, bx2), min(H - 1, by2)

                crop = frame[by1:by2, bx1:bx2]
                if crop.size == 0:
                    continue
                cls_name, sim, emb = self._match_class(crop) if self.prototypes else ("Unknown", 0.0, None)
                if sim < proto_sim_th:
                    continue

                detections.append({
                    "box": [bx1, by1, bx2, by2],
                    "score": float(sc),
                    "cls": cls_name,
                    "sim": float(sim),
                    "emb": emb
                })

        if not detections:
            return []
        boxes_arr = np.array([d["box"] for d in detections], dtype=np.int32)
        scores_arr = np.array([d["score"] * d["sim"] for d in detections], dtype=np.float32)
        keep_idx = nms_boxes(boxes_arr, scores_arr, iou_thresh=0.4)
        return [detections[i] for i in keep_idx]



class SimpleTracker:
    def __init__(self, max_age=15, iou_thresh=0.3, sim_thresh=0.7, lambda_sim=0.5):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.sim_thresh = sim_thresh
        self.lambda_sim = lambda_sim

    def update(self, dets):
        matched = set()
        for tid, track in list(self.tracks.items()):
            track["age"] += 1
            if track["age"] > self.max_age:
                del self.tracks[tid]
                continue
            best_score = 0
            best_det_idx = -1
            for i, d in enumerate(dets):
                if i in matched: continue
                iou = box_iou(track["box"], d["box"])
                sim = cos_sim(track["emb"], d["emb"]) if track["emb"] is not None and d["emb"] is not None else 0
                score = iou + self.lambda_sim * sim if sim > self.sim_thresh else iou
                if score > best_score and iou > self.iou_thresh:
                    best_score = score
                    best_det_idx = i
            if best_det_idx >= 0:
                det = dets[best_det_idx]
                track["box"] = det["box"]
                track["cls"] = det["cls"]
                track["sim"] = det["sim"]
                track["score"] = det["score"]
                track["emb"] = det["emb"]
                track["age"] = 0
                matched.add(best_det_idx)

        for i, d in enumerate(dets):
            if i not in matched:
                self.tracks[self.next_id] = {
                    "box": d["box"],
                    "cls": d["cls"],
                    "sim": d["sim"],
                    "score": d["score"],
                    "emb": d["emb"],
                    "age": 0
                }
                self.next_id += 1

        return [{"id": tid, **track} for tid, track in self.tracks.items()]

class YOLO_SAM2_Detector:
    def __init__(self, yolo_ckpt, sam2_ckpt, sam2_cfg):
        print("[INIT] Loading YOLOv11...")
        self.yolo = YOLO(yolo_ckpt)
        self.yolo.to(device)
        print("[INFO] YOLO loaded.")

        print("[INIT] Loading SAM2...")
        sd, info = load_sam2_weights(sam2_ckpt)
        self.classes = info.get("classes", [])
        self.model = build_sam2(sam2_cfg, None, device)
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.float().to(device).eval()
        self.predictor = SAM2ImagePredictor(self.model)
        print("[INFO] SAM2 loaded.")

        self.backbone = build_backbone(pretrained=True)
        self.prototypes = build_prototypes(PATHS["ANNOTATED_DIR"], self.classes, self.backbone, max_per_class=80)
        for k in list(self.prototypes.keys()):
            v = self.prototypes[k].ravel()
            self.prototypes[k] = v / (np.linalg.norm(v) + 1e-8)

    def _match_class(self, crop):
        if crop.size == 0:
            return "Unknown", 0.0, None
        inp = preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.backbone(inp).squeeze().cpu().numpy().ravel()
        qn = q / (np.linalg.norm(q) + 1e-8)
        best_cls, best_sc = "Unknown", -1
        for cls, proto in self.prototypes.items():
            sc = float(np.dot(qn, proto))
            if sc > best_sc:
                best_sc, best_cls = sc, cls
        return best_cls, best_sc, qn

    @torch.inference_mode()
    def infer_frame(self, frame, conf_thres=0.35, proto_sim_th=0.40):
        """
        YOLO detect → SAM2 refine mask (batch) → multi-object detection
        """
        detections = []

        yolo_results = self.yolo.predict(frame, conf=conf_thres, verbose=False)
        boxes = []
        for r in yolo_results:
            if len(r.boxes) == 0:
                continue
            xyxys = r.boxes.xyxy.cpu().numpy()
            boxes.extend([tuple(map(int, box)) for box in xyxys])
        if not boxes:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        boxes_np = np.array(boxes, dtype=np.int32)

        try:
            masks, scores, _ = self.predictor.predict(box=boxes_np, multimask_output=True)
        except Exception as e:
            print("[WARN] SAM2 batch inference failed:", e)
            return []

        if isinstance(masks, list):
            all_masks, all_scores = [], []
            for i, mset in enumerate(masks):
                if isinstance(mset, (list, np.ndarray)):
                    for j, m in enumerate(mset):
                        all_masks.append(np.array(m))
                        val = scores[i][j] if isinstance(scores[i], (list, np.ndarray)) else scores[i]
                        all_scores.append(float(val))
            masks = np.array(all_masks)
            scores = np.array(all_scores)

        if masks.ndim == 4:
            masks = masks[:, 0]
        H, W = frame.shape[:2]
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if isinstance(score, (list, np.ndarray)):
                score = float(np.max(score))
            else:
                score = float(score)
            if score < 0.30:
                continue

            mask = masks[i]
            if mask.ndim > 2:
                mask = mask[0]
            mask = mask.astype(np.uint8)

            if mask.sum() < 200:
                continue

            coords = np.argwhere(mask > 0)
            if coords.size == 0:
                continue
            ys, xs = coords[:, 0], coords[:, 1]
            bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(W - 1, bx2), min(H - 1, by2)

            crop = frame[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue

            cls_name, sim, emb = self._match_class(crop)
            if sim < proto_sim_th:
                continue

            detections.append({
                "box": [bx1, by1, bx2, by2],
                "score": float(score),
                "cls": cls_name,
                "sim": float(sim),
                "emb": emb
            })

        if not detections:
            return []

        boxes_arr = np.array([d["box"] for d in detections], dtype=np.int32)
        scores_arr = np.array([d["score"] * d["sim"] for d in detections], dtype=np.float32)
        keep_idx = nms_boxes(boxes_arr, scores_arr, iou_thresh=0.3)
        final_dets = [detections[i] for i in keep_idx]

        print(f"[INFO] Detected {len(final_dets)} objects.")
        return final_dets



def run_webcam_hybrid():
    print("[INFO] 🚀 Khởi tạo YOLO + SAM2 hybrid detector...")
    detector = YOLO_SAM2_Detector(
        yolo_ckpt=os.path.join(BASE_DIR, "checkpoints", "yolo11n.pt"),
        sam2_ckpt=PATHS["SAM2_CKPT"],
        sam2_cfg=PATHS["SAM2_CONFIG"]
    )
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[INFO] Webcam sẵn sàng — nhấn 'Q' để thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        start = time.time()

        dets = detector.infer_frame(frame, conf_thres=0.35, proto_sim_th=0.40)
        tracked = tracker.update(dets)

        out = frame.copy()
        for d in tracked:
            x1, y1, x2, y2 = map(int, d["box"])
            color = (int(37*d['id'] % 255), int(17*d['id'] % 255), int(233*d['id'] % 255))

            label = f"{d['cls']} {d['score']:.2f} sim:{d['sim']:.2f} id:{d['id']}"
            
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        fps = 1.0 / (time.time() - start + 1e-6)
        (h, w) = out.shape[:2]
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("YOLO + SAM2 Multi-Object Detection", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_hybrid()

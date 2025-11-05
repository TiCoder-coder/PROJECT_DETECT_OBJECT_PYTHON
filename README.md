PROJECT_DETECT_OBJECT — Real-time Object Recognition System (YOLOv11 + SAM2.1 + Classifier)

![Detect Object Preview](https://upload.wikimedia.org/wikipedia/commons/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg)

🚀 Overview

    Hệ thống Realtime Object Detection & Segmentation kết hợp nhiều mô hình AI mạnh mẽ:
        - YOLOv11 (Pretrained) — Dò tìm vật thể nhanh và chính xác.
        - SAM2.1 (Segment Anything 2) — Phân vùng chính xác (segmentation) từng vật thể được YOLO phát hiện.
        - ResNet18 (Custom Classifier) — Phân loại chi tiết từng vật thể dựa trên dữ liệu huấn luyện tùy chỉnh.
        - ImageSearcher (Embedding-based Similarity Search) — Khi xác suất thấp, hệ thống tìm vật thể tương tự trong thư viện annotated/.
        - Object Tracking + Label Stabilization — Theo dõi vật thể qua khung hình để tránh nhấp nháy nhãn.
        - Tất cả được xử lý real-time từ webcam, với giao diện hiển thị mask, bounding box, và tên vật thể ngay trên màn hình.

🏗️ System Architecture

1️⃣ Input Layer — Webcam Frame Capture
    - Luồng video lấy trực tiếp từ webcam (qua cv2.VideoCapture).
    - Mỗi frame được đưa vào hàng đợi (frame_queue) cho xử lý nền (thread).

2️⃣ YOLOv11 Detector
    - Model YOLOv11 pretrained (ultralytics.YOLO) xử lý detection nhanh chóng.
    - Xuất ra danh sách các bounding box [x1, y1, x2, y2].

3️⃣ SAM2.1 Segmenter
    - Dựa trên YOLO bounding boxes → SAM2.1 tạo segmentation mask chính xác cho từng vật thể.
    - Trọng số tùy chỉnh nạp từ: data/final_pth_to_webcam/sam2_inference_weights_latest.pth
    - File cấu hình: configs/sam2.1/sam2.1_hiera_b+.yaml
4️⃣ Custom Classifier (ResNet18 Fine-tuned)
    - Model ResNet18 được huấn luyện riêng trên dataset 102 lớp.
    - Checkpoint: /media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/data/final_pth_to_webcam/sam2_inference_weights_latest.pth
    - Khi phát hiện vật thể, phần ảnh được crop theo mask → phân loại qua classifier.

5️⃣ Image Searcher (Backup Matching)
    - Nếu độ tin cậy của classifier < 0.85, hệ thống tìm ảnh tương tự nhất trong thư viện data/annotated/ bằng cosine similarity giữa feature embedding.
6️⃣ Object Tracker

    - Theo dõi các bounding box qua khung hình (IOU-based tracking).

    - Làm mượt tọa độ và nhãn vật thể qua bbox_smooth_alpha.

    - Giúp nhãn không nhấp nháy khi camera di chuyển.

7️⃣ Display Layer

    - Hiển thị bounding box, mask (màu khác nhau) và label trực tiếp trên video.

    - FPS được tính theo thời gian thực.

    - Có thể dùng cv2.imshow hoặc fallback matplotlib nếu OpenCV không mở được cửa sổ.

📂 Folder Structure

PROJECT_DETECT_OBJECT/
## <!-- 
├── 📁 NOTEBOOK_TO_REPORT
│   ├── 📄 Analyst_accuracy_segement.ipynb
│   ├── 📄 Analyst_accuracy_yolo.ipynb
│   ├── 📄 automatic_mask_generator_example.ipynb
│   ├── 📄 image_predictor_example.ipynb
│   └── 📄 video_predictor_example.ipynb
├── 📁 configs
│   ├── 📁 sam2.1
│   │   ├── ⚙️ sam2.1_hiera_b+.yaml
│   │   ├── ⚙️ sam2.1_hiera_l.yaml
│   │   ├── ⚙️ sam2.1_hiera_s.yaml
│   │   ├── ⚙️ sam2.1_hiera_t.yaml
│   │   └── ⚙️ sam2.1_hiera_t.yaml.fixed.yaml.fixed.yaml
│   ├── 📁 sam2.1_training
│   │   └── ⚙️ sam2.1_hiera_b+_MOSE_finetune.yaml
│   └── 📁 yolo
│       └── ⚙️ yolo_learning_tools.yaml
├── 📁 sam2

│   ├── 📁 sam2
│   │   ├── 📁 csrc
│   │   │   └── 📄 connected_components.cu
│   │   ├── 📁 modeling
│   │   │   ├── 📁 backbones
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 hieradet.py
│   │   │   │   ├── 🐍 image_encoder.py
│   │   │   │   └── 🐍 utils.py
│   │   │   ├── 📁 sam
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 mask_decoder.py
│   │   │   │   ├── 🐍 prompt_encoder.py
│   │   │   │   └── 🐍 transformer.py
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 memory_attention.py
│   │   │   ├── 🐍 memory_encoder.py
│   │   │   ├── 🐍 position_encoding.py
│   │   │   ├── 🐍 sam2_base.py
│   │   │   └── 🐍 sam2_utils.py
│   │   ├── 📁 utils
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 amg.py
│   │   │   ├── 🐍 misc.py
│   │   │   └── 🐍 transforms.py
│   │   ├── ⚙️ _C.so
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 automatic_mask_generator.py
│   │   ├── 🐍 benchmark.py
│   │   ├── 🐍 build_sam.py
│   │   ├── ⚙️ sam2_hiera_b+.yaml
│   │   ├── ⚙️ sam2_hiera_l.yaml
│   │   ├── ⚙️ sam2_hiera_s.yaml
│   │   ├── ⚙️ sam2_hiera_t.yaml
│   │   ├── 🐍 sam2_image_predictor.py
│   │   ├── 📄 sam2_image_predictor.py.bak
│   │   ├── 🐍 sam2_train.py
│   │   ├── 🐍 sam2_video_predictor.py
│   │   └── 🐍 sam2_video_predictor_legacy.py
│   ├── 📁 tools
│   │   └── 🐍 vos_inference.py
│   ├── 📁 training
│   │   ├── 📁 assets
│   │   │   ├── 📄 MOSE_sample_train_list.txt
│   │   │   └── 📄 MOSE_sample_val_list.txt
│   │   ├── 📁 dataset
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 coco_raw_dataset.py
│   │   │   ├── 🐍 sam2_datasets.py
│   │   │   ├── 🐍 transforms.py
│   │   │   ├── 🐍 utils.py
│   │   │   ├── 🐍 vos_dataset.py
│   │   │   ├── 🐍 vos_raw_dataset.py
│   │   │   ├── 🐍 vos_sampler.py
│   │   │   └── 🐍 vos_segment_loader.py
│   │   ├── 📁 model
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 sam2.py
│   │   ├── 📁 modeling
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 scripts
│   │   │   └── 🐍 sav_frame_extraction_submitit.py
│   │   ├── 📁 utils
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 checkpoint_utils.py
│   │   │   ├── 🐍 data_utils.py
│   │   │   ├── 🐍 distributed.py
│   │   │   ├── 🐍 logger.py
│   │   │   ├── 🐍 misc.py
│   │   │   └── 🐍 train_utils.py
│   │   ├── 📝 README.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 loss_fns.py
│   │   ├── 🐍 optimizer.py
│   │   ├── 🐍 train.py
│   │   └── 🐍 trainer.py
│   ├── 🐍 __init__.py
│   ├── 📄 backend.Dockerfile
│   ├── ⚙️ docker-compose.yaml
│   ├── ⚙️ pyproject.toml
│   └── 🐍 setup.py
├── 📁 scripts
│   ├── 🐍 __init__.py
│   ├── 🐍 annote_data.py
│   ├── 🐍 create_ann.py
│   ├── 🐍 inference_webcam.py
│   ├── 🐍 merge_COCO.py
│   ├── 🐍 merge_LABEL.py
│   ├── 🐍 preprocess_data.py
│   ├── 🐍 reaname_file.py
│   └── 🐍 train.py ## -->

⚙️ Environment Setup

1️⃣ Create Environment
    cd PROJECT_DETECT_OBJECT
    python3 -m venv .venv
    source .venv/bin/activate

2️⃣ Install Dependencies
    pip install -r requirements.txt

3️⃣ Checkpoint Preparation
    | Model                     | Path                                                         | Description                    |
    | ------------------------- | ------------------------------------------------------------ | ------------------------------ |
    | **SAM2.1**                | `data/final_pth_to_webcam/sam2_inference_weights_latest.pth` | Custom finetuned SAM weights   |
    | **Classifier (ResNet18)** | `output/experiments/checkpoints/static_finetune_epoch12.pth` | Finetuned classification model |
    | **YOLOv11 Pretrained**    | `checkpoints/yolov11n.pt`                                    | Pretrained detection model     |
    | **Config**                | `configs/sam2.1/sam2.1_hiera_b+.yaml`                        | SAM2 architecture config       |

▶️ Run Real-time Detection
    python scripts/inference_webcam.py

🧩 Options

    - Press q to quit webcam window.

    - Modify cam_id if multiple cameras:
        inferencer.run(cam_id=1)
    - Adjust max_draw (number of displayed masks):
        inferencer = WebcamInferencer(..., max_draw=5)

💡 How the Pipeline Works Internally
    1. Capture Frame
        Reads image from webcam in a loop.

    2. Queue Handling
        Frame sent to inference_worker thread.

    3. YOLOv11 Inference
        Detects rough object bounding boxes.

    4. SAM2 Prediction
        Refines detection → pixel-level masks.

    5. Classifier + Image Searcher
        Assigns label using deep classification and similarity matching.

    5. Tracking
        Matches objects across frames using IoU.

    5. Display
        Draw masks, boxes, and names on live webcam feed.

🧠 Performance Notes

    - Uses multi-threading to separate webcam capture and AI inference.

    - Supports both CPU and GPU automatically (cuda or cpu).

    - Can handle ~10–15 FPS on RTX 3060 or similar GPU.

🧾 Logs & Debugging

| Level           | Prefix                             | Description |
| --------------- | ---------------------------------- | ----------- |
| `[INFO]`        | General system info                |             |
| `[WARN]`        | Missing files / fallback defaults  |             |
| `[SUCCESS]`     | Successful model or label loading  |             |
| `[FATAL ERROR]` | Critical load or inference failure |             |

🧩 Extensions
🔹 Replace YOLOv11 model checkpoint with custom trained weights.

🔹 Fine-tune SAM2.1 with custom masks dataset.

🔹 Add new annotated images for stronger Image Searcher performance.

🔹 Integrate SORT/ByteTrack for more stable multi-object tracking.

🎯 Summary
| Component     | Framework          | Purpose               |
| ------------- | ------------------ | --------------------- |
| YOLOv11       | Ultralytics        | Object Detection      |
| SAM2.1        | Meta FAIR          | Mask Segmentation     |
| ResNet18      | PyTorch            | Object Classification |
| ImageSearcher | Custom             | Similarity Matching   |
| Tracker       | Custom (IOU-based) | Temporal Stability    |

🖼️ Output Example
    When webcam runs successfully, you'll see:
        - Colored mask overlay per object
        - Bounding box with label name and confidence
        - Live FPS counter in terminal


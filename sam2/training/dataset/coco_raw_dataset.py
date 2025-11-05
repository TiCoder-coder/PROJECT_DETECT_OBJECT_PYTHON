import json, os
from training.dataset.vos_raw_dataset import VOSVideo, VOSFrame

class COCORawDataset:
    def __init__(self, coco_json, img_root):
        with open(coco_json, "r") as f:
            self.coco = json.load(f)
        self.img_root = img_root
        self.images = self.coco.get("images", [])
        self.annotations = self.coco.get("annotations", [])
        self.index_by_image = {img["id"]: img for img in self.images}

    def get_video(self, idx):
        img = self.images[idx]
        img_path = os.path.join(self.img_root, img["file_name"])
        frame = VOSFrame(frame_idx=img["id"], image_path=img_path)
        video = VOSVideo(video_name=img["file_name"], video_id=img["id"], frames=[frame])
        return video, self.annotations

    def __len__(self):
        return len(self.images)

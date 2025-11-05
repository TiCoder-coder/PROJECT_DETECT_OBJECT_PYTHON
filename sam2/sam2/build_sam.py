import logging
import os
import yaml

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def _load_config(config_file, overrides=None):
    """
    Load config from either full path (yaml file) or via hydra compose.
    returns OmegaConf config.
    """
    if os.path.isfile(config_file):
        with open(config_file, "r") as f:
            raw_cfg = yaml.safe_load(f)
        cfg = OmegaConf.create(raw_cfg)
        return cfg
    else:
        return compose(config_name=config_file, overrides=overrides or [])

def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    **kwargs,
):
    import sys
    from omegaconf import OmegaConf as _OmegaConf

    # YOLO
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sam2_root = os.path.abspath(os.path.join(current_dir, ".."))
    project_root = os.path.abspath(os.path.join(sam2_root, ".."))
    repo_root = os.path.abspath(os.path.join(project_root, ".."))

    for p in [repo_root, project_root, sam2_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # ANNOTE/ TRAIN 
    # import sam2.sam2.modeling as real_modeling

    # WEBCAM
    import sam2.modeling as real_modeling
    import sys as _sys
    _sys.modules["sam2.modeling"] = real_modeling 

    hydra_overrides_extra = hydra_overrides_extra or []
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    cfg = _load_config(config_file, overrides=hydra_overrides_extra)

    cfg_dict = _OmegaConf.to_container(cfg, resolve=False)

    def fix_targets(node):
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k == "_target_" and isinstance(v, str) and "sam2.sam2." in v:
                    new_v = v.replace("sam2.sam2.", "sam2.")
                    print(f"[DEBUG] Fixed target: {v} → {new_v}")
                    node[k] = new_v
                else:
                    fix_targets(v)
        elif isinstance(node, list):
            for item in node:
                fix_targets(item)

    fix_targets(cfg_dict)

    cfg = _OmegaConf.create(cfg_dict)
    _OmegaConf.resolve(cfg)

    print("[DEBUG] Final model target:", cfg.model._target_)

    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides_extra = hydra_overrides_extra or []
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor" ,
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor" ,
            "++model.compile_image_encoder=True",
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
    hydra_overrides = hydra_overrides + hydra_overrides_extra

    if os.path.isfile(config_file):
        with open(config_file, "r") as f:
            raw_cfg = yaml.safe_load(f)
        cfg = OmegaConf.create(raw_cfg)
        if vos_optimized:
            cfg.model._target_ = "sam2.sam2_video_predictor.SAM2VideoPredictorVOS"
            OmegaConf.update(cfg, "model.compile_image_encoder", True, merge=False)
        else:
            cfg.model._target_ = "sam2.sam2_video_predictor.SAM2VideoPredictor"
    else:
        cfg = compose(config_name=config_file, overrides=hydra_overrides)

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    """Load checkpoint smartly: combine pretrained weights and finetuned state_dict."""
    import torch
    import logging
    from collections import OrderedDict

    if ckpt_path is None:
        logging.info("No checkpoint provided, skipping load.")
        return
    if not os.path.isfile(ckpt_path):
        logging.warning(f"Checkpoint not found: {ckpt_path}. Skipping load.")
        return

    try:
        ck = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return

    # Detect format
    if isinstance(ck, dict):
        if "model_state_dict" in ck:
            sd_finetune = ck["model_state_dict"]
            logging.info("[INFO] Detected fine-tune checkpoint.")
        elif "model" in ck:
            sd_finetune = ck["model"]
            logging.info("[INFO] Detected nested 'model' key checkpoint.")
        else:
            sd_finetune = ck
    else:
        sd_finetune = ck

    # Load pretrained base weights (if exist in same folder)
    base_ckpt_path = os.path.join(os.path.dirname(ckpt_path), "sam2.1_hiera_base_plus.pt")
    sd_base = None
    if os.path.isfile(base_ckpt_path):
        try:
            sd_base = torch.load(base_ckpt_path, map_location="cpu")
            if isinstance(sd_base, dict) and "model" in sd_base:
                sd_base = sd_base["model"]
            logging.info(f"[INFO] Loaded base checkpoint: {base_ckpt_path}")
        except Exception as e:
            logging.warning(f"[WARN] Failed to load base weights: {e}")

    # Merge fine-tuned weights into base (if both available)
    if sd_base is not None:
        merged_sd = OrderedDict(sd_base)
        merged_sd.update({k.replace("module.", ""): v for k, v in sd_finetune.items()})
    else:
        merged_sd = OrderedDict({k.replace("module.", ""): v for k, v in sd_finetune.items()})

    # Load state dict with non-strict to allow partial match
    missing, unexpected = model.load_state_dict(merged_sd, strict=False)

    logging.warning(f"[WARN] Unexpected keys ignored ({len(unexpected)}): {unexpected[:5]}...")
    logging.warning(f"[WARN] Missing keys ignored ({len(missing)}): {missing[:5]}...")
    logging.info(f"[INFO] Checkpoint merged and loaded successfully ({len(merged_sd)} params).")

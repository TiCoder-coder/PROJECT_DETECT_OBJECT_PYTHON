import logging
import math
import os
import random
import re
from datetime import timedelta
from typing import Optional

import hydra

import numpy as np
import omegaconf
import torch
import torch.distributed as dist
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf


def multiply_all(*args):
    return np.prod(np.array(args)).item()


def collect_dict_keys(config):
    """This function recursively iterates through a dataset configuration, and collect all the dict_key that are defined"""
    val_keys = []
    # If the this config points to the collate function, then it has a key
    if "_target_" in config and re.match(r".*collate_fn.*", config["_target_"]):
        val_keys.append(config["dict_key"])
    else:
        # Recursively proceed
        for v in config.values():
            if isinstance(v, type(config)):
                val_keys.extend(collect_dict_keys(v))
            elif isinstance(v, omegaconf.listconfig.ListConfig):
                for item in v:
                    if isinstance(item, type(config)):
                        val_keys.extend(collect_dict_keys(item))
    return val_keys


class Phase:
    TRAIN = "train"
    VAL = "val"


def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    OmegaConf.register_new_resolver("get_class", hydra.utils.get_class)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("times", multiply_all)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
    OmegaConf.register_new_resolver("pow", lambda x, y: x**y)
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
    OmegaConf.register_new_resolver("range", lambda x: list(range(x)))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("ceil_int", lambda x: int(math.ceil(x)))
    OmegaConf.register_new_resolver("merge", lambda *x: OmegaConf.merge(*x))


def setup_distributed_backend(backend, timeout_mins):
    """
    Initialize torch.distributed.
    Tự động fallback về single-process mode nếu không có biến môi trường RANK/WORLD_SIZE.
    Tương thích với cả mô hình SAM2 single-GPU và multi-GPU.
    """
    import logging
    import torch.distributed as dist
    from datetime import timedelta

    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    logging.info(f"[Distributed] Setting up backend={backend} (timeout={timeout_mins} mins)")

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logging.warning("[Distributed] Env vars missing. Running in single-GPU mode (world_size=1).")

        # Thiết lập mặc định để code không bị crash
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))

        try:
            # Dù single GPU, vẫn init process group “nhẹ” để các hàm khác (barrier, all_reduce, v.v.) hoạt động
            dist.init_process_group(
                backend="gloo",  # gloo hoạt động tốt trên CPU cho single GPU
                init_method="tcp://127.0.0.1:12355",
                rank=0,
                world_size=1,
                timeout=timedelta(minutes=timeout_mins),
            )
            logging.info("[Distributed] Initialized local process group (rank=0, world_size=1).")
        except Exception as e:
            logging.warning(f"[Distributed] Local init failed ({e}), proceeding without distributed.")
            return 0

        return dist.get_rank()

    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_mins))
    logging.info("[Distributed] Multi-GPU process group initialized successfully.")
    return dist.get_rank()




def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current GPU.
    Safe fallback: defaults to (0, 0, 1) when env variables are missing.
    """
    import os

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    except (TypeError, ValueError):
        local_rank = 0

    try:
        distributed_rank = int(os.environ.get("RANK", 0))
    except (TypeError, ValueError):
        distributed_rank = 0

    try:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    except (TypeError, ValueError):
        world_size = 1

    # Debug log cho chắc
    import logging
    logging.info(f"[RANK DEBUG] LOCAL_RANK={local_rank}, RANK={distributed_rank}, WORLD_SIZE={world_size}")

    return local_rank, distributed_rank, world_size



def print_cfg(cfg):
    """
    Supports printing both Hydra DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    logging.info(OmegaConf.to_yaml(cfg))


def set_seeds(seed_value, max_epochs, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    # Since in the pytorch sampler, we increment the seed by 1 for every epoch.
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"MACHINE SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        logging.info(f"Error creating directory: {dir_path}")
    return is_success


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_amp_type(amp_type: Optional[str] = None):
    if amp_type is None:
        return None
    assert amp_type in ["bfloat16", "float16"], "Invalid Amp type."
    if amp_type == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float16


def log_env_variables():
    env_keys = sorted(list(os.environ.keys()))
    st = ""
    for k in env_keys:
        v = os.environ[k]
        st += f"{k}={v}\n"
    logging.info("Logging ENV_VARIABLES")
    logging.info(st)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MemMeter:
    """Computes and stores the current, avg, and max of peak Mem usage per iteration"""

    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0  # Per iteration max usage
        self.avg = 0  # Avg per iteration max usage
        self.peak = 0  # Peak usage for lifetime of program
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, n=1, reset_peak_usage=True):
        self.val = torch.cuda.max_memory_allocated() // 1e9
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        self.peak = max(self.peak, self.val)
        if reset_peak_usage:
            torch.cuda.reset_peak_memory_stats()

    def __str__(self):
        fmtstr = (
            "{name}: {val"
            + self.fmt
            + "} ({avg"
            + self.fmt
            + "}/{peak"
            + self.fmt
            + "})"
        )
        return fmtstr.format(**self.__dict__)


def human_readable_time(time_seconds):
    time = int(time_seconds)
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02}d {hours:02}h {minutes:02}m"


class DurationMeter:
    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def add(self, val):
        self.val += val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


class ProgressMeter:
    def __init__(self, num_batches, meters, real_meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch, enable_print=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [
            " | ".join(
                [
                    f"{os.path.join(name, subname)}: {val:.4f}"
                    for subname, val in meter.compute().items()
                ]
            )
            for name, meter in self.real_meters.items()
        ]
        logging.info(" | ".join(entries))
        if enable_print:
            print(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_resume_checkpoint(checkpoint_save_dir):
    if not g_pathmgr.isdir(checkpoint_save_dir):
        return None
    ckpt_file = os.path.join(checkpoint_save_dir, "checkpoint.pt")
    if not g_pathmgr.isfile(ckpt_file):
        return None

    return ckpt_file

import json
import re
import os
import random
import numpy as np
import torch
from datetime import datetime
from typing import Tuple, List


def get_device():
    """
    Detect and return the best available device for PyTorch.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU

    Returns:
        torch.device: The best available device
        str: Device name for logging ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "mps"
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    return device, device_name


def fix_seeds(seed):
    """
    Fix random seeds for reproducibility across different backends.
    Supports CUDA, MPS (Apple Silicon), and CPU.
    """
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)

    # CUDA-specific seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # MPS-specific seeds (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def setup_logging(log_root, log_header, dataset_name, model_ckpt, note):
    model_name = model_ckpt.split("/")[-1]
    assert "/" not in log_header
    assert "/" not in dataset_name
    assert "/" not in model_name
    assert "/" not in note
    log_dir = os.path.join(log_root, f'{log_header}_logs/[{note}] {dataset_name}_{model_name}/{datetime.now().strftime("%Y-%m%d-%H%M")}')
    os.makedirs(log_dir, exist_ok=True)   
    return log_dir


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
    return js


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    return txt


def read_raw_data_dir(raw_data_dir, recursive=True) -> List[str]:
    """only read txt files"""
    data = []
    if recursive:
        for root, dirs, files in os.walk(raw_data_dir):
            for f in files:
                if "txt" not in f:
                    continue
                full_path = os.path.join(root, f)
                d = read_txt(full_path)
                data.append(d)
    else:
        raise NotImplementedError
    
    return data
import argparse
import sys
import cv2
import numpy as np
import time
import threading
import psutil
import pynvml
from ptflops import get_model_complexity_info
import os

import torch
import torch.nn as nn

import mmcv
from embeded_fcm.model_wrappers.co_detr import CO_DINO_5scale_9encdoer_lsj_r50_3x_coco

from embeded_fcm.fcm.data.test_dataset import SFUHW

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Unsplit CO-DETR Inference")
    
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/workspace/datasets/SFU_HW_Obj"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint file path for use"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config file path for use"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory path for save results"
    )
    
    args = parser.parse_args(argv)
    return args

def test(model, args):
    device = args.device
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    model = model.to(device).eval()
    dataset_list = os.listdir(args.dataset)
    
    for data in dataset_list:
        print(data)

def main(argv):
    args = parse_args(argv)
    device = args.device
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model_config = args.config
    model_checkpoint = args.checkpoint
    model = CO_DINO_5scale_9encdoer_lsj_r50_3x_coco(
        device=device,
        model_config=model_config,
        model_checkpoint=model_checkpoint
    )
    
    test(model, args)

if __name__ == "__main__":
    main(sys.argv[1:])
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
import json
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mmcv
from embeded_fcm.model_wrappers.co_detr import CO_DINO_5scale_9encdoer_lsj_r50_3x_coco

from embeded_fcm.data.test_dataset import SFUHW

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
        dataset = SFUHW(
            root=os.path.join(args.dataset, data),
            annotation_file=f"annotations/{data}.json"
        )
        data_infos = dataset.load_annotations(dataset.annotation_path)
        dataset.data_infos = data_infos
        dataset._dataset = data_infos
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False,
            pin_memory=(device == "cuda"),
            collate_fn=lambda x:x
        )
        
        results = []
        for idx, batch in enumerate(dataloader):
            imgs = batch[0]['img']
            with torch.no_grad():
                result = model.unsplit(imgs, device)
            
            results.append(result)
            # for det in result:
            #     det_np = [d for d in det]
            #     results.append(det_np)
                
        eval_results = dataset.evaluate(
            results,
            metric='bbox'
        )
        
        # save results on json file
        
        # print results and complexity on terminal

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
    
    # set logger
    
    # print specs
    
    test(
        model,
        args
    )

if __name__ == "__main__":
    main(sys.argv[1:])
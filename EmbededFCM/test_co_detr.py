import torch
import torch.nn as nn

import cv2

from model_wrappers import CO_DINO_5scale_9encdoer_lsj_r50_3x_coco

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CO_DINO_5scale_9encdoer_lsj_r50_3x_coco(device="cpu", model_config="/workspace/EmbededFCM/models/CO-DETR/projects/configs/co_dino/co_dino_5scale_9encoder_lsj_r50_3x_coco.py", model_checkpoint="/workspace/EmbededFCM/checkpoints/co_dino_5scale_9encoder_lsj_r50_3x_coco.pth")
    
    # load image
    img = cv2.imread("/workspace/EmbededFCM/models/CO-DETR/demo/demo.jpg")
    
    features, metas = model.input_to_features(img, device)
    
    out = model.features_to_output(features, metas, device)
    
    print()
    
if __name__ == "__main__":
    main()
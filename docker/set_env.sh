#!/bin/bash

set -e

git clone https://github.com/Sense-X/Co-DETR.git

pip install --no-cache-dir --upgrade pip wheel setuptools
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html
pip install openmim timm
mim install mmdet==2.25.3

pip install fairscale==0.4.13 scipy==1.10.1 scikit-learn
pip install -U Cython

pip install yapf==0.40.1
pip install matplotlib numpy pycocotools six terminaltables fvcore tensorboard mmcv einops
import torch

from mmcv.parallel import collate

from model_wrappers.mmdet.apis import init_detector
from model_wrappers.mmdet.datasets import replace_ImageToTensor
from model_wrappers.mmdet.datasets.pipelines import Compose

from .base_wrappers import BaseWrapper
from .projects import *

class CO_DINO_5scale_9encdoer_lsj_r50_3x_coco(BaseWrapper):
    def __init__(self, device: str, model_config, model_checkpoint, **kwargs):
        super().__init__(device)
        
        self.model = init_detector(config=model_config, checkpoint=model_checkpoint, device=device)
        
        self.cfg = self.model.cfg
        
        self.backbone = self.model.backbone
        self.neck = self.model.neck
    
    def input_to_features(self, x, device):
        self.model = self.model.to(device).eval()
        
        return self._input_to_backbone(x)
    
    def features_to_output(self, x, device):
        pass
    
    def _input_to_backbone(self, x):
        with torch.no_grad():
            self.cfg.data.test.pipeline = replace_ImageToTensor(self.cfg.data.test.pipeline)
            test_pipeline = Compose(self.cfg.data.test.pipeline)
            
            data = dict(img=x)
            data = test_pipeline(data)
            
            data = collate(data, samples_per_gpu=1)
        
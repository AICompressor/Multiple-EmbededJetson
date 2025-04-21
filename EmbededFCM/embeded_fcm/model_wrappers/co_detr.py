import torch

from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from .mmdet.apis import init_detector
from .mmdet.datasets import replace_ImageToTensor
from .mmdet.datasets.pipelines import Compose
from .mmdet.core import bbox2result

from .base_wrappers import BaseWrapper
from .projects import *

class CO_DINO_5scale_9encdoer_lsj_r50_3x_coco(BaseWrapper):
    def __init__(self, device: str, model_config, model_checkpoint, **kwargs):
        super().__init__(device)
        
        self.model = init_detector(config=model_config, checkpoint=model_checkpoint, device=device)
        
        self.cfg = self.model.cfg
    
    def input_to_features(self, x, device):
        self.model = self.model.to(device).eval()
        
        return self._input_to_backbone(x, device)
    
    def features_to_output(self, x, metas, device):
        self.model = self.model.to(device).eval()
             
        results, x = self.model.query_head.simple_test(x, metas, rescale=True, return_encoder_output=True)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.model.query_head.num_classes)
            for det_bboxes, det_labels in results
        ]
        
        return bbox_results
    
    def _input_to_backbone(self, x, device):
        with torch.no_grad():
            self.cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            
            self.cfg.data.test.pipeline = replace_ImageToTensor(self.cfg.data.test.pipeline)
            test_pipeline = Compose(self.cfg.data.test.pipeline)
            
            data = dict(img=x)
            data = test_pipeline(data)
            
            data['img_metas'] = [img_metas.data for img_metas in data['img_metas']]
            data['img'] = [img.data for img in data['img']]
            
            if next(self.model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                for m in self.model.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'
            
            data['img_metas'][0]['batch_input_shape'] = data['img'][0].size()[1:]
            return self.model.extract_feat(data['img'][0].unsqueeze(0)), data['img_metas']
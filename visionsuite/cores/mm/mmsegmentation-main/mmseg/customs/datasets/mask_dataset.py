#!/usr/bin/env python3
import os.path as osp
import os
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

# @DATASETS.register_module()
# class MaskDataset(BaseSegDataset):
#     def __init__(
#         self,
#         classes,
#         img_suffix,
#         seg_map_suffix,
#         **kwargs,
#     ):
#         self.CLASSES = classes
#         super(MaskDataset, self).__init__(**kwargs)
        
#         self.img_suffix = img_suffix
#         self.seg_map_suffix = seg_map_suffix
#         self.img_infos = self.load_annotations(
#             img_dir=self.img_dir,
#             ann_dir=self.ann_dir,
#         )

#     def load_annotations(self, img_dir, ann_dir ):
#         img_infos = []
#         for img_name in os.listdir(img_dir):
#             if img_name.endswith(self.img_suffix):
#                 img_info = dict(filename=img_name)
#                 img_info['ann'] = dict(seg_map=osp.join(ann_dir, img_name))
#                 img_infos.append(img_info)
        
#         return img_infos
        
#     def pre_pipeline(self, results):
#         results["seg_fields"] = []
#         results['img_prefix'] = self.img_dir
#         results['seg_prefix'] = self.ann_dir

# # Copyright (c) OpenMMLab. All rights reserved.
# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MaskDataset(BaseSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    # METAINFO = dict(
    #     palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230]])
    METAINFO = dict()
    def __init__(self,
                 classes,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        
        self.METAINFO.update({'classes': classes})
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import numpy as np
import imgviz
import json
import mmcv
import pandas as pd
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test#, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes
import visionsuite.cores.spatial_transform_decoupling.mmrotate_custom.datasets.custom_dota_dataset as custom_dota_dataset
from visionsuite.utils.visualizers.vis_obb import vis_obb
from visionsuite.utils.helpers import JsonEncoder



def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default='/HDD/etc/std/Spatial-Transform-Decoupling/src/configs/rotated_imted/dota/vit/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_dota_ms_rr_le90_stdc_xyawh321v.py')
    parser.add_argument('--checkpoint', default='/HDD/_projects/github/VisionSuite/visionsuite/cores/spatial_transform_decoupling/work_dirs/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_dota_ms_rr_le90_stdc_xyawh321v/epoch_300.pth')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        default=[0, 1],
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', default='/HDD/etc/std/test_300')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    # ~/anaconda3/envs/openmmlab/lib/python3.7/site-packages/mmdet/models/dense_heads/anchor_head.py:123: UserWarning: 
    # DeprecationWarning: anchor_generator is deprecated, please use "prior_generator" instead
    #   warnings.warn('DeprecationWarning: anchor_generator is deprecated, '
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if 'samples_per_gpu' in cfg.data.test:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            test_dataloader_default_args['samples_per_gpu'] = \
                cfg.data.test.pop('samples_per_gpu')
        if test_dataloader_default_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            if 'samples_per_gpu' in ds_cfg:
                warnings.warn('`samples_per_gpu` in `test` field of '
                              'data will be deprecated, you should'
                              ' move it to `test_dataloader` field')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        test_dataloader_default_args['samples_per_gpu'] = samples_per_gpu
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    class2idx = {'BOX': 0}
    idx2class = {0: 'BOX'}
    output_dir = '/HDD/_projects/benchmark/obb_detection/rich/tests/std_hivit'
    json_dir = '/HDD/_projects/benchmark/obb_detection/rich/datasets/split_dataset_box/val'
    iou_threshold = 0.2
    
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    compare_gt = True
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]

    model.eval()
    dataset = data_loader.dataset
    idx2xyxys = {}
    compare = {}
    results = {}
    for i, data in enumerate(data_loader):
        
        img_metas = data['img_metas'][0].data[0]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
        for idx, res in enumerate(result):
            labels = [
                np.full(bbox[0].shape[0], i, dtype=int)
                for i, bbox in enumerate([res])
            ]
            labels = np.concatenate(labels)
            bboxes = np.vstack(res)
            
            _idx2xyxys = {}
            for i, (label, bbox) in enumerate(zip(labels, bboxes)):
                
                xc, yc, w, h, ag = bbox[:5]
                wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
                hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
                p1 = [xc - wx - hx, yc - wy - hy]
                p2 = [xc + wx - hx, yc + wy - hy]
                p3 = [xc + wx + hx, yc + wy + hy]
                p4 = [xc - wx + hx, yc - wy + hy]
                poly = [p1, p2, p3, p4]
                
                if label not in _idx2xyxys:
                    _idx2xyxys[int(label)] = {'polygon': [poly], 'confidence': [bbox[-1]]}
                else:
                    _idx2xyxys[int(label)]['polygon'].append(poly)
                    _idx2xyxys[int(label)]['confidence'].append(bbox[-1])
                    
            
        img_file = img_metas[idx]['filename']
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        results.update({filename: {'idx2xyxys': _idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
            
        _compare = vis_obb(img_file, _idx2xyxys, idx2class, output_dir, color_map, json_dir, 
                           compare_gt=compare_gt, iou_threshold=iou_threshold)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})


    with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4, cls=JsonEncoder)

    if compare_gt:
        with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
            json.dump(compare, json_file, ensure_ascii=False, indent=4)
        
        df_compare = pd.DataFrame(compare)
        df_compare_pixel = df_compare.loc['diff_iou'].T
        df_compare_pixel.fillna(0, inplace=True)
        df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))

    # if not distributed:
    #     model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    #     outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
    #                               args.show_score_thr)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    # rank, _ = get_dist_info()
    # if rank == 0:
    #     if args.out:
    #         print(f'\nwriting results to {args.out}')
    #         mmcv.dump(outputs, args.out)
    #     kwargs = {} if args.eval_options is None else args.eval_options
    #     if args.format_only:
    #         dataset.format_results(outputs, **kwargs)
    #     if args.eval:
    #         eval_kwargs = cfg.get('evaluation', {}).copy()
    #         # hard-code way to remove EvalHook args
    #         for key in [
    #                 'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
    #                 'rule', 'dynamic_intervals'
    #         ]:
    #             eval_kwargs.pop(key, None)
    #         eval_kwargs.update(dict(metric=args.eval, **kwargs))
    #         metric = dataset.evaluate(outputs, **eval_kwargs)
    #         print(metric)
    #         metric_dict = dict(config=args.config, metric=metric)
    #         if args.work_dir is not None and rank == 0:
    #             mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()

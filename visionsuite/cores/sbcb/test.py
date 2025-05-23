#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import numpy as np
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.utils import DictAction
from mmseg.utils import setup_multi_processes
from mmseg.apis import multi_gpu_test, single_gpu_test

from bbseg import digit_version
from bbseg.datasets import build_dataloader, build_dataset
from bbseg.models import build_segmentor
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="mmseg test (and eval) a model")
    # parser.add_argument("--config", default='/HDD/_projects/github/VisionSuite/visionsuite/cores/sbcb/configs/deeplabv3plus/deeplabv3plus_r50b-d8_512x1024_40k_cityscapes.py')
    parser.add_argument("--config", default='/HDD/_projects/github/VisionSuite/visionsuite/cores/sbcb/configs/deeplabv3plus/deeplabv3plus_r101b-d8_512_512_mask.py')
    parser.add_argument("--checkpoint", default='/HDD/_projects/benchmark/semantic_segmentation/new_model/outputs/sbcb/20241125/epoch_200.pth', help="checkpoint file")
    parser.add_argument(
        "--work-dir", default='/HDD/_projects/benchmark/semantic_segmentation/new_model/tests/sbcb_241125',
        help=(
            "if specified, the evaluation metric results will be dumped"
            "into the directory as json"
        ),
    )
    parser.add_argument(
        "--aug-test", action="store_true", help="Use Flip and Multi scale aug"
    )
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server (note: forces `test` split)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True, 
        help="Uses test split instead of validation split",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes',
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show results",
    )
    parser.add_argument(
        "--show-dir", 
        default='/HDD/_projects/benchmark/semantic_segmentation/new_model/tests/sbcb_241125',
        help="directory where painted images will be saved",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed testing)",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu_collect is not specified",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        "not be supported in version v0.22.0. Override some settings in the "
        "used config, the key-value pair in xxx=yyy format will be merged "
        "into config file. If the value to be overwritten is a list, it "
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        "marks are necessary and that no white space is allowed.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=1,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both "
            "specified, --options is deprecated in favor of --cfg-options. "
            "--options will not be supported in version v0.22.0."
        )
    if args.options:
        warnings.warn(
            "--options is deprecated in favor of --cfg-options. "
            "--options will not be supported in version v0.22.0."
        )
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    # Test or Validation split
    if args.test:
        use_val = False
    else:
        use_val = True
    # if args.format_only:  # NOTE: we need validation for evaluating with SegFix!
    #     use_val = False

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # Select config (test or val)
    if use_val:
        cfg.data.test = cfg.data.val

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
        ]  # added 2.0
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f"The gpu-ids is reset from {cfg.gpu_ids} to "
                f"{cfg.gpu_ids[0:1]} to avoid potential error in "
                "non-distribute testing time."
            )
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir, f"eval_multi_scale_{timestamp}.json")
        else:
            json_file = osp.join(args.work_dir, f"eval_single_scale_{timestamp}.json")
    elif rank == 0:
        work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir, f"eval_multi_scale_{timestamp}.json")
        else:
            json_file = osp.join(work_dir, f"eval_single_scale_{timestamp}.json")
        args.work_dir = work_dir  # set work_dir as args

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    if not torch.cuda.is_available():
        assert digit_version(mmcv.__version__) >= digit_version(
            "1.4.4"
        ), "Please use MMCV >= 1.4.4 for CPU training!"
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    
    
    # results = single_gpu_test(
    #     model,
    #     data_loader,
    #     show=args.show,
    #     out_dir=args.show_dir,
    #     efficient_test=False,
    #     opacity=args.opacity,
    #     pre_eval=args.eval is not None and not eval_on_format_results,
    #     format_only=args.format_only or eval_on_format_results,
    #     format_args=eval_kwargs,
    # )

    from mmcv.image import tensor2imgs
    import imgviz

    color_map = imgviz.label_colormap(50)
    pre_eval = True
    model.eval()
    results = []
    dataset = data_loader.dataset
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)


        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for img, img_meta, pred in zip(imgs, img_metas, result):
            h, w, _ = img_meta['img_shape']
            vis_img = np.zeros((h, w*2, 3))
            vis_img[:, w:, :] = color_map[pred]
            vis_img[:, :w, :] = img
            
            cv2.imwrite(osp.join(args.show_dir, img_meta['ori_filename']), vis_img)
            

            # ori_h, ori_w = img_meta['ori_shape'][:-1]
            # img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            # if args.show_dir:
            #     out_file = osp.join(args.show_dir, img_meta['ori_filename'])
            # else:
            #     out_file = None

            # model.module.show_result(
            #     img_show,
            #     result,
            #     palette=dataset.PALETTE,
            #     show=show,
            #     out_file=out_file,
            #     opacity=opacity)

        # if format_only:
        #     result = dataset.format_results(
        #         result, indices=batch_indices, **format_args)
        # if pre_eval:
        #     # TODO: adapt samples_per_gpu > 1.
        #     # only samples_per_gpu=1 valid now
        #     result = dataset.pre_eval(result, indices=batch_indices)
        #     results.extend(result)
        # else:
        #     results.extend(result)


    # return results


if __name__ == "__main__":
    main()

_base_ = [
    '../../../_base_/datasets/rich_dota.py',
    '../../../_base_/default_runtime.py'
    # '../../../_base_/schedules/schedule_1x_adamw.py'
]

pretrained = '/HDD/weights/std/mae_pretrain_vit_base_full.pth'

angle_version = 'le90'
norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='RotatedimTED',
    # pretrained=pretrained,
    proposals_dim=6,
    backbone=dict(
        type='VisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.2,
        learnable_pos_embed=True,
        use_checkpoint=True,
        with_simple_fpn=True,
        last_feat=True),
    neck=dict(
        type='SimpleFPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        norm_cfg=norm_cfg,
        use_residual=False,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_skip_fpn=False,
    with_mfm=True,
    roi_head=dict(
        type='OrientedStandardRoIHeadimTED',
        bbox_roi_extractor=[dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=768,
            featmap_strides=[4, 8, 16, 32]),
                            dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=768,
            featmap_strides=[16])],
        bbox_head=dict(
            type='RotatedMAEBBoxHead',
            init_cfg=dict(type='Pretrained', checkpoint=pretrained),
            use_checkpoint=True,
            in_channels=768,
            img_size=224,
            patch_size=16, 
            embed_dim=512, 
            depth=8,
            num_heads=16, 
            mlp_ratio=4., 
            # reg_decoded_bbox=True,
            # 以下参数照抄Oriented RCNN ##### #####
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # 以下cfg照抄Oriented RCNN ##### #####
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))

fp16 = dict(loss_scale=dict(init_scale=512))
device = 'cuda'

# from default_runtime.py -----------------------------------
workflow = [('train', 1), ('val', 10)]
# ------------------------------------------------------------

# from schedulers --------------------------------------------
# evaluation
evaluation = dict(interval=300, metric='mAP')

# optimizer
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor', 
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.75))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=10, create_symlink=False)
# ------------------------------------------------------------

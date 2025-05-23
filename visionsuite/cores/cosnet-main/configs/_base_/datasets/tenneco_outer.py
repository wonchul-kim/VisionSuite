dataset_type = 'MaskDataset'
data_root = "/HDD/datasets/projects/Tenneco/Metalbearing/outer/241120/split_patch_mask_dataset"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 1120)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='Resize', img_scale=crop_size, keep_ratio=True),#ratio_range=(0.8, 1.2)),
    # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            #dict(type='AlignResize', keep_ratio=True, size_divisor=32),
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/masks',
        classes=('background', 'line', 'stabbed'),
        img_suffix='.bmp',
        seg_map_suffix='.bmp',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/masks',
        classes=('background', 'line', 'stabbed'),
        img_suffix='.bmp',
        seg_map_suffix='.bmp',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='./',
        ann_dir='./',
        classes=('background', 'line', 'stabbed'),
        img_suffix='.bmp',
        seg_map_suffix='.bmp',
        pipeline=test_pipeline))
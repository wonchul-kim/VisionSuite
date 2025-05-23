_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/zero_waste.py',
    '../_base_/default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='/cosnet/pretrain/model_best.pth.tar',  # pretrain (imagenet) weight path 
    backbone=dict(
        type='COSNet',
        depths=[3, 3, 12, 3],
        style='pytorch'),
    decode_head=dict(num_classes=5,
                     in_channels=[72, 72*2, 72*4, 72*8],
                     channels=256,
                     in_index=[0, 1, 2, 3],
                     norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=5,
                        in_channels=72*4,
                        in_index=2,         #in_index=4,
                        norm_cfg=norm_cfg)
    )



gpu_multiples = 1  # we used 1 gpu
# optimizer
optimizer = dict(type='AdamW', lr=0.00009*gpu_multiples, betas=(0.9, 0.999), weight_decay=0.001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', warmup='linear', warmup_iters=1500,
                 warmup_ratio=1e-6, power=0.95, min_lr=1e-7, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=4000//gpu_multiples)
evaluation = dict(interval=4000//gpu_multiples, metric='mIoU', save_best='mIoU')

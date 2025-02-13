_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py', '../../_base_/datasets/lx.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_20k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4))

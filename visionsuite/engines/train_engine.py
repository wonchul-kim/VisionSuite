import torch
import os.path as osp


input_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets'
classes = ['scratch', 'tear', 'stabbed']

device_ids = '0'
batch_size = 1
num_workers = 0
drop_last=True

device = torch.device(f'cuda:{device_ids.split(",")[0]}')

from visionsuite.engines.segmentation.datasets.mask_dataset import MaskDataset
dataset = MaskDataset(classes, input_dir, imgs_dir_name='patches')

from torch.utils.data.dataloader import DataLoader

dataloader = DataLoader(dataset, 
                        batch_size=batch_size,
                        num_workers=num_workers,
                        # collate_fn=collate_fn,
                        drop_last=True,
                    )

from visionsuite.engines.segmentation.models.unet3plus.UNet_3Plus import UNet_3Plus
from visionsuite.engines.segmentation.models.unet3plus.losses.iouLoss import IOU_loss

model = UNet_3Plus(n_classes=len(classes) + 1)
model.to(device)
model.train()


for batch in dataloader:
    input_image, input_target = batch['input'].to(device, dtype=torch.float32), batch['target'].to(device, dtype=torch.float32)
    print("input_image, input_target: ", input_image.shape, input_target.shape)
    pred = model(input_image)
    print("> pred: ", pred.shape)
    
    
    
    
    
    
    







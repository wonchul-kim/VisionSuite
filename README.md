# VisionSuite

This is just personal repository to easily use oepn source algorithms.

----------------------------------------------------------------------------------------------
## How to publish github-repository to PYPI

```cmd
git add .
git commit -m 'publish v1.x.x'
git tag v.1.x.x
git push origin --tags
```

----------------------------------------------------------------------------------------------
## Cores

### Roboflow 

#### Ultralytics

- HBB Detection

- OBB Detection
 
- Instance Segmentation

### Spatial Transform Decoupling

- OBB Detection

### Masked AutoEncoder

- 


----------------------------------------------------------------------------------------------
## Utils

### Configs 

This is for UI by `pydantic`

### Augmentation

### dataset

#### converters

- dota
    - dota2yolo

- labelme
    - dota
        - labelme2dota
    
    - labelme2yolo
        - labelme2yolo_hbb
        - labelme2yolo_is
        - labelme2yolo_obb

- coco

### loggers


----------------------------------------------------------------------------------------------
# Engines

## Classification

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="3">PyTorch</td>   | ResNet (torcivision)      | o           | o           | o           | -        |  |
|                                | Vision Transformer (ViT)  | x           | x           | x           | -        |  |
|                                | EfficientNet              | x           | x           | x           | -        |  |


## Detection

### HBB Detection

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="5">PyTorch</td>   | Yolov5                    | o           | o           | o           | -        | anchor-based                                                                |
|                                | Yolov7                    | x           | x           | x           | -        | anchor-based                                                                |
|                                | Yolov8                    | x           | x           | x           | -        | anchor-free                                                                 |
|                                | DETR                      | x           | x           | x           | -        |                                                                             |
|                                | RTDETR                    | x           | x           | x           | -        | transformer                                                                 |

### OBB Detection

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="3">PyTorch</td>   | RTMDet                    | o           | o           | o           | -        |  |
|                                |  | x           | x           | x           | -        |  |
|                                |  | x           | x           | x           | -        |  |

## Segmentation

### Semantic Segmentation

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="3">PyTorch</td>   | DeepLabV3+                | o           | o           | o           | -        |  |
|                                | Mask2Former               | x           | x           | x           | -        |  |
|                                | OneFormer                 | x           | x           | x           | -        |  |

### Instance Segmentation

| | Library                      |            Model          |     train   |    test     |    export   | ImageNet |    Note                                                                     |
|-|:----------------------------:|:-------------------------:|:-----------:|:-----------:|:-----------:|:--------:|:---------------------------------------------------------------------------:|
| <td rowspan="3">PyTorch</td>   | Mask RCNN (torchvision)   | o           | o           | o           | -        |  
|                                |    | x           | -      | x           | x           | x           | -        |  
|                                |    | x           | -      | x           | x           | x           | -        |  
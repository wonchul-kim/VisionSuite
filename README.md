# VisionSuite

This is just personal repository to easily use oepn source algorithms.

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



## How to publish github-repository to PYPI

```cmd
git add .
git commit -m 'publish v1.x.x'
git tag v.1.x.x
git push origin --tags
```
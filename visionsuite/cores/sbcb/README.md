# Boosting Semantic Segmentation with Semantic Boundaries

This repository contains the code for our paper "Boosting Semantic Segmentation with Semantic Boundaries".

```
@article{ishikawa2023SBDboost,
  title={Boosting Semantic Segmentation with Semantic Boundaries},
  author={Ishikawa, Haruya and Aoki Yoshimitsu},
  journal={Sensors},
  year={2023},
  volume={23},
  pages={6980},
  doi={https://doi.org/10.3390/s23156980}
}
@article{ishikawa2023SBDboost_preprint,
  title={Boosting Semantic Segmentation with Semantic Boundaries},
  author={Ishikawa, Haruya and Aoki Yoshimitsu},
  journal={arXiv preprint arXiv: http://arxiv.org/abs/2304.09427},
  year={2023}
}
```
## Guides

- [Installation/Setup](.readme/installation.md)
  - If you are not used to the `mmseg` ecosystem, these guides might helpful
- [How to apply SBCB for your own model](.readme/sbcb_model.md)
  - Guides for applying the SBCB framework to your own model
- [Reproducing results for used in our paper](.readme/reproduce.md)
  - Guides for reproducing the results in our paper
  - Currently supports DeepLabV3, DeepLabV3+, and PSPNet
  - 4/27: Also added configs for SegFormer


### HOW TO TRAIN

#### Set the environment

1. Download docker image
  ```
  docker build -t mmcv1.6.1-torch1.8.1-cuda11.1-cudnn8 https://github.com/open-mmlab/mmcv.git#master:docker/release --build-arg MMCV=1.6.1 --build-arg PYTORCH=1.8.1 --build-arg CUDA=11.1 --build-arg CUDNN=8
  ```

2. Run docker image

3. Install libraries
  ```
  pip install mmsegmentation==0.27.0 scikit-image scikit-learn pyEdgeEval yapf==0.40.1
  ```


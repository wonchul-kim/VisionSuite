## Set environment to run mm

```
docker build -t mmcv2.1.0 https://github.com/open-mmlab/mmcv.git#master:docker/release --build-arg PYTORCH=1.8.0 --build-arg CUDA=11.1 --build-arg CUDNN=8
```
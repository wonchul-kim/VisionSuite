#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train.py --launcher pytorch ${@:3}

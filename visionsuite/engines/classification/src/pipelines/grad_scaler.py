import torch
from visionsuite.engines.classification.utils.registry import PIPELINES

@PIPELINES.register()
def get_scaler(amp):
    if amp:
        return torch.cuda.amp.GradScaler()
    else:
        return None
    
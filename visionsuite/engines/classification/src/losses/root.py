
import torch.nn as nn 
from visionsuite.engines.classification.utils.registry import LOSSES

@LOSSES.register()
def cross_entropy(loss_config):
    return nn.CrossEntropyLoss(label_smoothing=loss_config['label_smoothing'])


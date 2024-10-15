import torch 
import torch.nn as nn

def criterion(inputs, target):
    losses = {}
    if isinstance(inputs, torch.Tensor):
        losses['out'] = nn.functional.cross_entropy(inputs, target, ignore_index=255)
    else:
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]
# import inspect
# from typing import List

# import tensorflow as tf

# from aivcommon.registry import LOSSES, Registry


# def register_tf_losses() -> List[str]:
#     TF_LOSSES = Registry("tf_loss", parent=LOSSES, scope="aivcommon.losses.tensorflow")
#     for module_name in dir(tf.keras.losses):
#         if module_name.startswith("__"):
#             continue
#         _object = getattr(tf.keras.losses, module_name)
#         if inspect.isclass(_object):  # and issubclass(_object, tf.keras.losses.losses):
#             TF_LOSSES.register(module=_object)

#     return TF_LOSSES


# TF_LOSSES = register_tf_losses()


import inspect
from typing import List

import torch
from visionsuite.engines.utils.registry import LOSSES, Registry


def register_torch_losses() -> List[str]:
    TORCH_LOSSES = Registry("torch_loss", parent=LOSSES, scope="aivcommon.losses.pytorch")
    for module_name in dir(torch.nn):
        if module_name.startswith("__"):
            continue
        
        if module_name.endswith("Loss"):
            _object = getattr(torch.nn, module_name)
        else:
            continue
        if inspect.isclass(_object):  # Check if it's a class
            # Optionally check if it's a subclass of nn.Module or specific loss classes
            # if issubclass(_object, torch.nn.Module): 
            TORCH_LOSSES.register(module=_object)

    return TORCH_LOSSES


TORCH_LOSSES = register_torch_losses()

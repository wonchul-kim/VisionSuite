import torch
from visionsuite.engines.classification.utils.registry import OPTIMIZERS
from visionsuite.engines.utils.torch_utils.utils import set_weight_decay

@OPTIMIZERS.register()
def get_parameters(bias_weight_decay, transformer_embedding_decay, model, weight_decay, norm_weight_decay):
    custom_keys_weight_decay = []
    if bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", bias_weight_decay))
    if transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, transformer_embedding_decay))
            
    parameters = set_weight_decay(
        model,
        weight_decay,
        norm_weight_decay=norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    return parameters
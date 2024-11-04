from visionsuite.engines.classification.utils.registry import LOSSES
from visionsuite.engines.utils.helpers import get_params_from_obj

@LOSSES.register()
def loss(loss_config):
    loss_obj = LOSSES.get(loss_config['type'])
    loss_params = get_params_from_obj(loss_obj)
    for key in loss_params.keys():
        if key in loss_config:
            loss_params[key] = loss_config[key]
        
    return loss_obj(**loss_params)

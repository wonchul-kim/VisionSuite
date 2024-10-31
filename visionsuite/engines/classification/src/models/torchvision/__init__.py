import torchvision
from visionsuite.engines.utils.registry import MODELS


@MODELS.register()
def torchvision_model(**config):
    try:
        return torchvision.models.get_model(name=config['model_name'] + config['backbone'], num_classes=config['num_classes'], weights=config['weights'])
    except Exception as error:
        raise RuntimeError(f"{error}: There has been error when loading torchvision model: {config['model_name']} with config({config}): ")
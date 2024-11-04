from visionsuite.engines.classification.utils.registry import MODELS

def build_model(args):
    model = MODELS.get(f"{args['model']['backend'].capitalize()}Model")()
    
    return model

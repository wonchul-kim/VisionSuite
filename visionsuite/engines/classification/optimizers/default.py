import torch

def get_optimizer(opt, lr, parameters, momentum, weight_decay):

    if opt.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt,
        )
    elif opt == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {opt}. Only SGD, RMSprop and AdamW are supported.")
    
    
    return optimizer
import torch

def get_optimizer(args, params_to_optimize):
    if args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=args['lr'])
    elif args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize, lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    else:
        raise NotImplementedError(f"There is no such optimizer: {args['optimizer']}")


    return optimizer
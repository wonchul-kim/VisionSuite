from visionsuite.engines.classification.utils.registry import DATASETS

           
@DATASETS.register()
def torchivision_image_folder_dataset(input_dir, transform):
    import torchvision
    
    return torchvision.datasets.ImageFolder(input_dir, transform)

# def load_data(traindir, valdir, transform, args):
#     import os
#     import torch
#     import torchvision
#     import time
#     import torchvision.transforms
#     from visionsuite.engines.utils.helpers import mkdir, get_cache_path
#     from visionsuite.engines.utils.torch_utils.utils import save_on_master
#     from visionsuite.engines.classification.utils.registry import SAMPLERS

#     # Data loading code
#     print("Loading data")
    

#     st = time.time()
#     cache_path = get_cache_path(traindir)
#     if args.cache_dataset and os.path.exists(cache_path):
#         # Attention, as the transforms are also cached!
#         print(f"Loading dataset_train from {cache_path}")
#         # TODO: this could probably be weights_only=True
#         dataset, _ = torch.load(cache_path, weights_only=False)
#     else:
#         dataset = torchvision.datasets.ImageFolder(
#             traindir,
#             transform,
#         )
#         if args.cache_dataset:
#             print(f"Saving dataset_train to {cache_path}")
#             mkdir(os.path.dirname(cache_path))
#             save_on_master((dataset, traindir), cache_path)
#     print("Took", time.time() - st)

#     print("Loading validation data")
#     cache_path = get_cache_path(valdir)
#     if args.cache_dataset and os.path.exists(cache_path):
#         # Attention, as the transforms are also cached!
#         print(f"Loading dataset_test from {cache_path}")
#         # TODO: this could probably be weights_only=True
#         dataset_test, _ = torch.load(cache_path, weights_only=False)
#     else:
#         if args.model['weights']:
#             weights = torchvision.models.get_weight(args.model['weights'])
#             preprocessing = weights.transforms(antialias=True)
#             if args.backend == "tensor":
#                 preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

#         else:
#             preprocessing = transform

#         dataset_test = torchvision.datasets.ImageFolder(
#             valdir,
#             preprocessing,
#         )
#         if args.cache_dataset:
#             print(f"Saving dataset_test to {cache_path}")
#             mkdir(os.path.dirname(cache_path))
#             save_on_master((dataset_test, valdir), cache_path)

#     print("Creating data loaders")


#     train_sampler, test_sampler = SAMPLERS.get('get_samplers')(args, dataset, dataset_test)

#     return dataset, dataset_test, train_sampler, test_sampler


import torch
import os.path as osp
from typing import Dict
import numpy as np
import random
from tqdm import tqdm
import gc

from datasets.data_utils import get_dataloaders
from models.dinov2_from_hunggingface import Dinov2FromHuggingFace
from visionsuite.engines.data.embeddings.utils.features import *
from visionsuite.engines.utils.torch_utils.torch_dist_env import enable_distributed, synchronize, get_global_rank, cleanup

import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist


class EmbeddingGenerator:
    def __init__(self, config):
        
        self._config = config
        
        self._set()
        self.set_model()
        self.set_dataset()
        
    @property 
    def config(self):
        return self._config 
    
    def _set(self):
        self._set_seed()
        self._set_dist()
        
    def _set_seed(self):
        if 'seed' not in self._config:
            seed = 42
        else:
            seed = self._config['seed']
            
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
    def _set_dist(self, use_torchrun=False, set_cuda_current_device: bool = True, overwrite: bool = False):
        enable_distributed(
            use_torchrun=use_torchrun,
            set_cuda_current_device=set_cuda_current_device,
            overwrite=overwrite,
        )
        
    def set_dataset(self):
        self._dataloader = get_dataloaders(config['dataset_format'], self._preprocess, 
                                     config['batch_size'], config['input_dir'],
                                     roi=config['roi'], search_all=config['search_all'])

    def set_model(self):
        
        if 'dinov2' in config['model_name']:
            self._model = Dinov2FromHuggingFace(self._config['output_dir'], self._config['model_name'],
                                                output_attentions=True if self._config['features']['type'] == 'attention' else False)
            self._preprocess = None
        # elif 'clip' in config['model_name']:
        #     ckpt_dir = os.path.join(args.root_dir, "checkpoints/clip")
        #     model, preprocess = clip.load(phi_to_name[args.phis], device=device, download_root=ckpt_dir)
        #     model.eval()
        #     print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        #     model = model.encode_image
        #     preprocess.transforms[2] = _convert_image_to_rgb
        #     preprocess.transforms[3] = _safe_to_tensor
        
        
        else:
            raise NotImplementedError(f"{self._config['model_name']} has not yet been considered")

    def get_features_from_huggingface(self):
        all_features = []
        filenames = []
        cnt = 1
        rank = get_global_rank()
        with torch.no_grad():
            for x, filename in tqdm(self._dataloader):
                ori_features = self._model(pixel_values=x.to(f'cuda:{rank}'))
                cls_embedding = ori_features.last_hidden_state[:, 0, :]      # [CLS] token embedding
                patch_embeddings = ori_features.last_hidden_state[:, 1:, :]  # patch-level embeddings
                attn_list = ori_features.attentions  # List[Tensor], one per layer > each attention: (bs, num_heads, num_tokens, num_tokens)
                patch_feats = patch_embeddings
                
                ### Patch-level Clustering
                if 'features' in self._config and self._config['features']:
                    if self._config['features']['type'] == 'clustering':
                        out_features = cluster_features(patch_feats, self._config['features'])
                        
                    elif self._config['features']['type'] == 'l2-norm':
                        out_features = l2_norm_features(patch_feats)
                        
                    elif self._config['features']['type'] == 'topk':
                        out_features = topk_features(patch_feats)
                    
                    elif self._config['features']['type'] == 'attention':
                        out_features = attention_feature(patch_feats, attn_list)
                        
                    else:
                        raise NotImplementedError(f"There is no such features method: {self._config['features']['type']}")
                        
                    if 'cat_cls_token' in self._config['features'] and self._config['features']['cat_cls_token']:
                        out_features = torch.cat([cls_embedding.cpu(), out_features], dim=-1)
                else:
                    out_features = out_features.detach().cpu()
    
                all_features.append(out_features)
                filenames.extend(filename)
                
                if len(all_features) != 0 and len(all_features) == self._config['save_embedding_number']:
                    
                    feats_train, filenames_train = torch.cat(all_features).numpy(), filenames

                    if 'features' in self._config and self._config['features']:
                        if self._config['features']['type'] == 'clustering':
                            representations_dir = f"{self._config['output_dir']}/{self._config['model_name']}/{self._config['features']['type']}_{self._config['features']['k']}_{self._config['features']['cat_cls_token']}"
                        else:
                            representations_dir = f"{self._config['output_dir']}/{self._config['model_name']}/{self._config['features']['type']}_{self._config['features']['cat_cls_token']}"
                            
                    if not os.path.exists(representations_dir):
                        os.makedirs(representations_dir)

                    np.save(f"{representations_dir}/train_rank_{rank}_{cnt}.npy", feats_train)

                    with open(f"{representations_dir}/train_filenames_rank_{rank}_{cnt}.txt", 'w') as f:
                        for path in filenames_train:
                            f.write(f"{path}\n")
                            
                    all_features.clear()
                    filenames.clear()
                cnt += 1
                
            feats_train, filenames_train = torch.cat(all_features).numpy(), filenames

            if 'features' in self._config and self._config['features']:
                if self._config['features']['type'] == 'clustering':
                    representations_dir = f"{self._config['output_dir']}/{self._config['model_name']}/{self._config['features']['type']}_{self._config['features']['k']}_{self._config['features']['cat_cls_token']}"
                else:
                    representations_dir = f"{self._config['output_dir']}/{self._config['model_name']}/{self._config['features']['type']}_{self._config['features']['cat_cls_token']}"

            if not os.path.exists(representations_dir):
                os.makedirs(representations_dir)

            np.save(f"{representations_dir}/train_rank_{rank}_{cnt}.npy", feats_train)
            synchronize()

            with open(f"{representations_dir}/train_filenames_rank_{rank}_{cnt}.txt", 'w') as f:
                for path in filenames_train:
                    f.write(f"{path}\n")
                    
            all_features.clear()
            filenames.clear()
            print(f"FINSHED RANK ({rank}) .......!!!!!!!!!!!!!!!!!!!!!")
            synchronize()
            
       
    def run(self):
        self.get_features_from_huggingface()
        print("=======================================>>> DONE")
        # gc.collect()
        # del self._dataloader
        # torch.cuda.empty_cache()
        synchronize()

if __name__ == '__main__':

    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    config = {'dataset_format': 'labelme', 
            # 'model_name': 'dinov2_vitb14',
            'model_name': 'dinov2-large',
            'batch_size': 1,
            'input_dir': '/HDD/etc/curation/unit/data',
            'output_dir': '/HDD/etc/curation/unit/embedding',
            'search_all': False,
            'device': 'cuda',
            'seed': 42,
            'roi': [],
            'features': 
                # {
                #     'type': 'clustering',
                #     'k': 50,
                #     'cat_cls_token': True
                # } 
                {
                    'type': 'attention',
                    'cat_cls_token': False
                },
            'save_embedding_number': 10000
            
        }
        
    
    # config = {'dataset_format': 'labelme', 
    #         # 'model_name': 'dinov2_vitb14',
    #         'model_name': 'dinov2-base',
    #         'batch_size': 1,
    #         'input_dir': '/HDD/datasets/projects/benchmarks/tenneco/split_dataset/train',
    #         'search_all': False,
    #         'output_dir': '/HDD/datasets/projects/benchmarks/tenneco/split_embedding_dataset',
    #         'device': 'cuda',
    #         'seed': 42,
    #         'roi': [220, 60, 1340, 828],
    #         'feature': 
    #             # {
    #             #     'type': 'clustering',
    #             #     'k': 50,
    #             #     'cat_cls_token': True
    #             # } 
    #             {
    #                 'type': 'attention',
    #                 'cat_cls_token': False
    #             } 
    #     }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    emb_generator = EmbeddingGenerator(config)
    emb_generator.run()
    print('dkdkdddddddddddddddddddddd')
    print("awlkefjawlkefjawklefjklawejf")
    synchronize()
    cleanup()
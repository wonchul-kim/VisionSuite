import torch
import os.path as osp
import numpy as np
import random
from tqdm import tqdm

from datasets.data_utils import get_dataloaders
from models.dinov2 import Dinov2FromFacebook
from models.dinov2_from_hunggingface import Dinov2FromHuggingFace


import matplotlib.pyplot as plt
import numpy as np

    
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
        
        self._device = torch.device(self._config['device'])
            
    
    def set_dataset(self):
        self._dataloader = get_dataloaders(config['dataset_format'], self._preprocess, 
                                     config['batch_size'], config['input_dir'],
                                     roi=config['roi'])

    def set_model(self):
        
        if 'dinov2' in config['model_name']:
            # self._model = Dinov2FromFacebook(self._config['output_dir'], self._config['model_name'])
            # self._preprocess = None
            self._model = Dinov2FromHuggingFace(self._config['output_dir'], self._config['model_name'],
                                                output_attentions=True if self._config['post']['type'] == 'attention' else False)
            # self._preprocess = self._model._processor
            self._preprocess = None
            
            # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
            # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
            
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
        with torch.no_grad():
            for x, filename in tqdm(self._dataloader):
                # x['pixel_values'] = x['pixel_values'][0].to(self._device)
                # features = self._model(**x)
                features = self._model(pixel_values=x.to(self._device))
                cls_embedding = features.last_hidden_state[:, 0, :]      # [CLS] token embedding
                patch_embeddings = features.last_hidden_state[:, 1:, :]  # patch-level embeddings


                attn_list = features.attentions  # List[Tensor], one per layer
                patch_feats = patch_embeddings
                
                ### Patch-level Clustering
                if 'post' in self._config and self._config['post']:
                    if self._config['post']['type'] == 'clustering':
                        from sklearn.cluster import KMeans
                        clustered_feats = []
                        for i in range(patch_feats.shape[0]):
                            # (4400, 768) → (K, 768)
                            kmeans = KMeans(n_clusters=self._config['post']['k']).fit(patch_feats[i].cpu().numpy())
                            centers = torch.tensor(kmeans.cluster_centers_)
                            clustered_feats.append(centers)  # shape: (5, 768)
                            
                        features = torch.stack(clustered_feats, dim=0).sum(dim=1)
                        
                    elif self._config['post']['type'] == 'l2-norm':
                        ### object-aware weighted pooling
                        ### 1. L2-norm
                        norms = patch_feats.norm(dim=-1)  # (B, N)
                        weights = torch.nn.functional.softmax(norms, dim=1) # (B, N)
                        features = (patch_feats * weights.unsqueeze(-1)).sum(dim=1).cpu().detach()  # (B, D)
                        
                    elif self._config['post']['type'] == 'topk':
                        ### 2. saliency-like: top-k patch pooling 
                        topk = 100
                        norms = patch_feats.norm(dim=-1)  # (B, N)
                        values, indices = torch.topk(norms, topk, dim=1)  # (B, topk)
                        pooled = []
                        for i in range(patch_feats.shape[0]):
                            top_feats = patch_feats[i][indices[i]]  # (topk, 768)
                            pooled.append(top_feats.mean(dim=0))    # (768,)
                        features = torch.stack(pooled, dim=0).cpu().detach()    # (B, 768)
                    
                    elif self._config['post']['type'] == 'attention':
                        ### 3. self-attention 
                        # 현재 이미지의 CLS→patch attention
                        attn = attn_list[-1]                 # (4401, 768)
                        cls_attn = attn[:, :, 0, 1:]         # shape: (B, heads, tokens, tokens)   
                        avg_attn = cls_attn.mean(dim=1)      
                        features = (patch_feats * avg_attn.unsqueeze(-1)).sum(dim=1).cpu().detach()  # (1, D)
                        
                    else:
                        raise NotImplementedError(f"There is no such post method: {self._config['post']['type']}")
                        
                        
                    if 'cat_cls_token' in self._config['post'] and self._config['post']['cat_cls_token']:
                        features = torch.cat([cls_embedding.cpu(), features], dim=-1)
                        
                        
                else:
                    features = features.detach().cpu()
    
                all_features.append(features)
                filenames.extend(filename)

        return torch.cat(all_features).numpy(), filenames
        
        
    def get_features_from_meta(self):
        all_features = []
        filenames = []
        with torch.no_grad():
            for x, filename in tqdm(self._dataloader):
                self.attn_maps.clear()                
                features = self._model(x.to(self._device))
                
                patch_feats = features["x_norm_patchtokens"]  # (B, N, D)
                
                ### Patch-level Clustering
                from sklearn.cluster import KMeans
                clustered_feats = []
                for i in range(patch_feats.shape[0]):
                    # (4400, 768) → (K, 768)
                    kmeans = KMeans(n_clusters=5).fit(patch_feats[i].cpu().numpy())
                    centers = torch.tensor(kmeans.cluster_centers_)
                    clustered_feats.append(centers)  # shape: (5, 768)

                                
                ### object-aware weighted pooling
                ### 1. L2-norm
                patch_feats = features["x_norm_patchtokens"]  # (B, N, D)
                norms = patch_feats.norm(dim=-1)  # (B, N)
                weights = torch.nn.functional.softmax(norms, dim=1) # (B, N)
                weighted_avg = (patch_feats * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

                ### 2. saliency-like: top-k patch pooling 
                topk = 100
                patch_feats = features["x_norm_patchtokens"]  # (B, N, D)
                norms = patch_feats.norm(dim=-1)  # (B, N)
                values, indices = torch.topk(norms, topk, dim=1)  # (B, topk)
                pooled = []
                for i in range(patch_feats.shape[0]):
                    top_feats = patch_feats[i][indices[i]]  # (topk, 768)
                    pooled.append(top_feats.mean(dim=0))    # (768,)
                pooled_feats = torch.stack(pooled, dim=0)    # (B, 768)

                                
                ### 3. self-attention 
                # 현재 이미지의 CLS→patch attention
                attn = self.attn_maps[0]                    # (4401, 768)
                cls_attn = attn[:, :, 0, 1:]           
                avg_attn = cls_attn.mean(dim=1)        
                # attention_features.append(avg_attn.cpu())  # 저장
                                            
                                
                # features = self._model(x.to(self._device), cls_token=True)
                all_features.append(features.detach().cpu())
                filenames.extend(filename)

        return torch.cat(all_features).numpy(), filenames
        
    def run(self):
        
        # feats_train, filenames_train = self.get_features_from_meta()
        feats_train, filenames_train = self.get_features_from_huggingface()

        if 'post' in self._config and self._config['post']:
            if self._config['post']['type'] == 'clustering':
                representations_dir = f"{self._config['output_dir']}/{self._config['model_name']}/{self._config['post']['type']}_{self._config['post']['k']}"
            else:
                representations_dir = f"{self._config['output_dir']}/{self._config['model_name']}/{self._config['post']['type']}"
                
        if not os.path.exists(representations_dir):
            os.makedirs(representations_dir)

        np.save(f"{representations_dir}/train.npy", feats_train)

        with open(f"{representations_dir}/train_filenames.txt", 'w') as f:
            for path in filenames_train:
                f.write(f"{path}\n")


if __name__ == '__main__':
    import os
    
    
    config = {'dataset_format': 'labelme', 
            # 'model_name': 'dinov2_vitb14',
            'model_name': 'dinov2-large',
            'batch_size': 1,
            'input_dir': '/HDD/datasets/projects/benchmarks/mr/split_patch_dataset/train',
            'output_dir': '/HDD/datasets/projects/benchmarks/mr/split_patch_embedding_dataset',
            'device': 'cuda',
            'seed': 42,
            'roi': [],
            'post': 
                # {
                #     'type': 'clustering',
                #     'k': 50,
                #     'cat_cls_token': True
                # } 
                {
                    'type': 'attention',
                    'cat_cls_token': True
                } 
        }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    emb_generator = EmbeddingGenerator(config)
    emb_generator.run()
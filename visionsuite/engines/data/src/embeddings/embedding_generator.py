import torch
import os.path as osp
import numpy as np
import random
from tqdm import tqdm

from visionsuite.engines.data.src.embeddings.datasets.data_utils import get_dataloaders


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
        
        if config['model_name'] == 'dinov2':
            self._model = torch.hub.set_dir(osp.join(config['output_dir'], "checkpoints/dinov2"))
            self._model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(self._device)
            self._model.eval()
            print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self._model.parameters()]):,}")
            self._preprocess = None
            
            # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
            # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
            
        # else:
        #     ckpt_dir = os.path.join(args.root_dir, "checkpoints/clip")
        #     model, preprocess = clip.load(phi_to_name[args.phis], device=device, download_root=ckpt_dir)
        #     model.eval()
        #     print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        #     model = model.encode_image
        #     preprocess.transforms[2] = _convert_image_to_rgb
        #     preprocess.transforms[3] = _safe_to_tensor
        
    def get_features(self):
        all_features = []
        filenames = []
        with torch.no_grad():
            for x, filename in tqdm(self._dataloader):
                features = self._model(x.to(self._device))
                all_features.append(features.detach().cpu())
                filenames.extend(filename)

        return torch.cat(all_features).numpy(), filenames
        
    def run(self):
        
        feats_train, filenames_train = self.get_features()

        representations_dir = f"{config['output_dir']}/representations/{config['model_name']}"
        if not os.path.exists(representations_dir):
            os.makedirs(representations_dir)

        np.save(f"{representations_dir}/{config['dataset_format']}_train.npy", feats_train)

        with open(f"{representations_dir}/{config['dataset_format']}_train_filenames.txt", 'w') as f:
            for path in filenames_train:
                f.write(f"{path}\n")


        
    
    
if __name__ == '__main__':
    import os
    
    config = {'dataset_format': 'labelme', 
            'model_name': 'dinov2',
            'batch_size': 4,
            'input_dir': '/HDD/etc/curation/tenneco/unit/data',
            'output_dir': '/HDD/etc/curation/tenneco/unit/embedding',
            'device': 'cuda',
            'seed': 42,
            'roi': [220, 60, 1340, 828]
        }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    emb_generator = EmbeddingGenerator(config)
    emb_generator.run()
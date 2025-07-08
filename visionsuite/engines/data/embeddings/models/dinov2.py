import torch 
import os.path as osp
import numpy as np


class Dinov2:
    def __init__(self, output_dir, model_name, device='cuda'):

        self._model = torch.hub.set_dir(osp.join(output_dir, "checkpoints"))
        self._model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        self._model.eval()
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self._model.parameters()]):,}")
        
        
    def infer(self, x, masks=None, full_return=True, cls_token=False, patch_token=False):
        
        output = self._model.forward_features(x, masks=masks)
        
        if full_return or (cls_token and patch_token):
            return output
        else:
            if cls_token:
                return output['x_norm_clstoken']
            
            if patch_token:
                return output['x_norm_patchtokens']

        
        
        
        
if __name__ == '__main__':
    
    output_dir = '/HDD/etc/outputs/embeddings/dinov2'
    model_name = 'dinov2_vitb14'
    
    model = Dinov2(output_dir, model_name)
    
    
    img_file = '/HDD/etc/curation/tenneco/unit/data/0_0_124062721060032_6_Outer.bmp'
    from torchvision import transforms
    from PIL import Image
    import torch

    image = Image.open(img_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((770, 1120)),
        # transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # same as DINOv2
                            std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to('cuda')  # shape: (1, 3, 518, 518)
    print("Input tensor shape: ", input_tensor.shape)

    with torch.no_grad():
        last_hidden = model._model.forward_features(input_tensor)
        
    cls_token = last_hidden['x_norm_clstoken']
    patch_token = last_hidden['x_norm_patchtokens']
    reg_token = last_hidden['x_norm_regtokens']
    pre_norm = last_hidden['x_prenorm']
    masks = last_hidden['masks']
    
    print("CLS Token Shape:", cls_token.shape)
    print("Patch Tokens Shape:", patch_token.shape)
    print("Reg Tokens Shape:", reg_token.shape)
    print("Pre Norm Shape:", pre_norm.shape)
    # print("Masks Shape:", masks.shape)
    
    import torch.nn.functional as F
    mean_patch = patch_token.mean(dim=1)
    similarity = F.cosine_similarity(cls_token, mean_patch)
    print("CLS vs Mean Patch similarity:", similarity.item())
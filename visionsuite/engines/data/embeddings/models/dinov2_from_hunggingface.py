import torch 
import os.path as osp
import numpy as np
from transformers import AutoModel, AutoImageProcessor

class Dinov2FromHuggingFace:
    def __init__(self, output_dir, model_name, device='cuda'):

        self._processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", 
                                                             do_resize=False,
                                                             do_center_crop=False)
        self._model = AutoModel.from_pretrained(f"facebook/{model_name}", output_attentions=False).to(device)
        self._model.eval()
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self._model.parameters()]):,}")
        
        
    def preprocess(self, image, return_tensors):
        '''
            def forward(
                        self,
                        pixel_values: Optional[torch.Tensor] = None,
                        bool_masked_pos: Optional[torch.Tensor] = None,
                        head_mask: Optional[torch.Tensor] = None,
                        output_attentions: Optional[bool] = None,
                        output_hidden_states: Optional[bool] = None,
                        return_dict: Optional[bool] = None,
                    ) -> Union[tuple, BaseModelOutputWithPooling]:
        
                    ...
                    ...
                return BaseModelOutputWithPooling(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
        '''
        return self._processor(images=image, return_tensors=return_tensors)
        
    def __call__(self, **x):
        
        return self._model(**x)        
        
        
if __name__ == '__main__':
    
    output_dir = '/HDD/etc/outputs/embeddings/dinov2'
    model_name = 'dinov2-base'
    model = Dinov2FromHuggingFace(output_dir, model_name)
    
    
    img_file = '/HDD/etc/curation/tenneco/unit/data/0_0_124062721060032_6_Outer.bmp'
    from PIL import Image
    import torch

    image = Image.open(img_file).convert("RGB")
    inputs = model.preprocess(image=image, return_tensors="pt").to('cuda')

    print("Input tensor: ", inputs)


    outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :]      # [CLS] token embedding
    patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # patch-level embeddings


    attn_list = outputs.attentions  # List[Tensor], one per layer
    last_attn = attn_list[-1]       # shape: (B, heads, tokens, tokens)

    # 5. [CLS] → patch attention 평균
    cls_attn = last_attn[:, :, 0, 1:]         # (1, heads, N_patches)
    avg_attn = cls_attn.mean(dim=1)           # (1, N_patches)

    # 6. 시각화 (예: 14x14 grid for ViT-B/14)
    import matplotlib.pyplot as plt

    def show_attention_map(attn_vector, grid_size=16):
        attn_map = attn_vector.reshape(grid_size, grid_size).cpu().detach().numpy()
        plt.imshow(attn_map, cmap='hot')
        plt.colorbar()
        plt.title("CLS → Patch Attention (mean over heads)")
        plt.axis('off')
        plt.show()

    show_attention_map(avg_attn[0])
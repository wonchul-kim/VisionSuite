import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.apis import init_model, inference_model
import dinov2.eval.segmentation.models

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

# Register all modules in mmsegmentation
register_all_modules()


def create_segmenter(cfg, backbone_model):
    # Initialize model from configuration
    # model = MODELS.build(cfg.model)
    model = init_model(cfg)
    # Replace backbone forward method with custom method
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model

# Load pretrained backbone =======================================================
# ================================================================================
BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()
print(backbone_model)


# Load pretrained segmentation head =======================================================
# ================================================================================
import urllib

import mmcv


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
HEAD_TYPE = "ms" # in ("ms, "linear")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = Config.fromstring(cfg_str, file_format=".py")
if HEAD_TYPE == "ms":
    cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
    print("scales:", cfg.data.test.pipeline[1]["img_ratios"])
del cfg.test_pipeline[1]
cfg.test_pipeline.append(dict(type='PackSegInputs'))
model = create_segmenter(cfg, backbone_model=backbone_model)
load_checkpoint(model, head_checkpoint_url, map_location="cpu")
model.cuda()
model.eval()
print(model)

# Load sample image =======================================================
# ================================================================================
import urllib

from PIL import Image


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"


image = load_image_from_url(EXAMPLE_IMAGE_URL)
image.save("/HDD/etc/ori.png")


# Semantic segmentation on sample image =======================================================
# ================================================================================]
import numpy as np

import dinov2.eval.segmentation.utils.colormaps as colormaps


DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}


def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1][0]
    return Image.fromarray(segmentation_values)


array = np.array(image)[:, :, ::-1] # BGR
# segmentation_logits = inference_segmentor(model, array)[0]
# segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)
# Preprocess image for the model
arr = array.copy()
img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()  # Convert to Tensor and add batch dimension
img_tensor = img_tensor / 255.0  # Normalize if needed
img_tensor = img_tensor.cuda()

with torch.no_grad():
    # result = inference_model(model, array)
    result = model.predict(img_tensor)

segmentation_logits = result[0]

from mmseg.apis import show_result_pyplot
segmented_image = render_segmentation(segmentation_logits.pred_sem_seg.data.cpu().detach(), HEAD_DATASET)

segmented_image.save("/HDD/etc/segmented_image.png")

# Load pretrained segmentation model (Mask2Former) =======================================================
# ================================================================================]
import dinov2.eval.segmentation_m2f.models.segmentors

CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

cfg_str = load_config_from_url(CONFIG_URL)
cfg = Config.fromstring(cfg_str, file_format=".py")

model = MODELS.build(cfg.model)
load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
model.cuda()
model.eval()

# Semantic segmentation on sample image =======================================================
# ================================================================================]
import numpy as np
array = np.array(image)[:, :, ::-1] # BGR
# segmentation_logits = inference_segmentor(model, array)[0]
# segmented_image = render_segmentation(segmentation_logits, "ade20k")
# segmented_image.save('/HDD/etc/segmented_image_m2f.png')

# Preprocess image for the model
img_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()  # Convert to Tensor and add batch dimension
img_tensor = img_tensor / 255.0  # Normalize if needed

# Forward pass through the model
with torch.no_grad():
    result = model.simple_test(img_tensor)

# Post-process the result (e.g., obtain logits, segmentation map)
segmentation_logits = result[0]['pred_sem_seg']  # Adjust based on output format

from mmseg.apis import show_result_pyplot
segmented_image = show_result_pyplot(
    image, segmentation_logits, cfg, palette='ade20k', show=False
)

# Save segmented image
segmented_image.save('/HDD/etc/segmented_image_m2f.png')
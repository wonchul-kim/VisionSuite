from visionsuite.engines.segmentation.datasets.coco_utils import get_coco
from visionsuite.engines.segmentation.datasets.mask_dataset import get_mask
import torchvision


def get_dataset(args, is_train):
    def sbd(*args, **kwargs):
        kwargs.pop("use_v2")
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    def voc(*args, **kwargs):
        kwargs.pop("use_v2")
        return torchvision.datasets.VOCSegmentation(*args, **kwargs)

    paths = {
        "voc": (args.input_dir, voc, 21),
        "voc_aug": (args.input_dir, sbd, 21),
        "coco": (args.input_dir, get_coco, 21),
        "mask": (args.input_dir, get_mask, 4),
    }
    p, ds_fn, num_classes = paths[args.dataset]

    image_set = "train" if is_train else "val"
    ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train, args), use_v2=args.use_v2)
    return ds, num_classes

from torchvision.transforms import functional as F, InterpolationMode
import visionsuite.engines.utils.torch_utils.presets as presets

def get_transform(is_train, args):
    if is_train:
        return presets.SegmentationPresetTrain(base_size=512, crop_size=480, backend=args.backend, use_v2=args.use_v2)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return presets.SegmentationPresetEval(base_size=512, backend=args.backend, use_v2=args.use_v2)
    
"""
Layer-by-layer features from two matched ImageNet-1k models.

Both models come from timm with explicit weight tags so the pretraining data
is the same (ImageNet-1k only). The old code compared torchvision ViT-B/16
(86M parameters) with ResNet-34 (22M), which is not a matched pair.

Every stage of each model is returned, flattened over space and tokens, so the
encoding analysis can sweep layers. For mouse visual cortex the best layer is
early to intermediate (Conwell et al. 2021); the final layer, which the old
code used, is the worst choice for V1.
"""

from typing import Dict, List

import numpy as np
import timm
import torch

WEIGHTS = {
    "vit_small": "vit_small_patch16_224.augreg_in1k",
    "resnet34": "resnet34.a1_in1k",
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_models(device: str) -> Dict[str, torch.nn.Module]:
    models = {}
    for name, tag in WEIGHTS.items():
        models[name] = timm.create_model(tag, pretrained=True).to(device).eval()
    return models


def layer_modules(name: str, model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """Named stages to record, in depth order."""
    if name == "resnet34":
        stages = {"stem": model.maxpool}
        for i in (1, 2, 3, 4):
            stages[f"layer{i}"] = getattr(model, f"layer{i}")
        stages["pool"] = model.global_pool
        return stages
    if name == "vit_small":
        stages = {"patch_embed": model.patch_embed}
        for i, block in enumerate(model.blocks):
            stages[f"block{i:02d}"] = block
        stages["norm"] = model.norm
        return stages
    raise ValueError(name)


@torch.no_grad()
def extract_all_layers(name: str, model: torch.nn.Module, images: np.ndarray, device: str,
                       batch_size: int = 16) -> Dict[str, np.ndarray]:
    """Features for every stage. Returns {stage: (n_images, n_features)} float32."""
    stages = layer_modules(name, model)
    captured: Dict[str, List[np.ndarray]] = {k: [] for k in stages}
    hooks = []
    for stage_name, module in stages.items():
        def make_hook(key):
            def hook(_module, _inp, out):
                captured[key].append(out.detach().flatten(1).float().cpu().numpy())
            return hook
        hooks.append(module.register_forward_hook(make_hook(stage_name)))

    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    for i in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[i:i + batch_size]).permute(0, 3, 1, 2).to(device)
        model((batch - mean) / std)
    for h in hooks:
        h.remove()
    return {k: np.concatenate(v, axis=0) for k, v in captured.items()}

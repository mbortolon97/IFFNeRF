import torch

def create_backbone(
    type="dino",
    pretrained=False,
    filter_size=4,
    pool_only=True,
    _force_nonfinetuned=False,
    **kwargs,
):
    if type == "dino":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        wh = (16, 16)
        num_features = 384
    return model, wh, num_features
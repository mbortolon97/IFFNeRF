import torch
from models.tensorBase import positional_encoding

class RayPreprocessor(torch.nn.Module):
    def __init__(self, viewpe=8, pospe=8, rgbpe=6, featureC=128, fea_output=128):
        super().__init__()

        self.in_mlpC = 2 * viewpe * 3 + 3 + 2 * pospe * 3 + 3 + 2 * rgbpe * 3 + 3
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC + self.in_mlpC, featureC)
        layer4 = torch.nn.Linear(featureC, fea_output)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
        )
        self.mlp2 = torch.nn.Sequential(
            layer3,
            torch.nn.ReLU(inplace=True),
            layer4,
        )
        self.viewpe = viewpe
        self.pospe = pospe
        self.rgbpe = rgbpe

    def forward(self, pts, viewdirs, rgb):
        indata = [pts, viewdirs, rgb]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        if self.rgbpe > 0:
            indata += [positional_encoding(rgb, self.rgbpe)]
        mlp_in = torch.cat(indata, dim=-1)
        first_block_result = self.mlp(mlp_in)
        return self.mlp2(torch.cat((first_block_result, mlp_in), dim=-1))

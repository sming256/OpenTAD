import torch.nn as nn
from einops.layers.torch import Rearrange


class NormModule(nn.Module):
    def __init__(self, norm_type, norm_dim, **kwargs):  # ["BN,"GN","LN"]
        super().__init__()

        assert norm_type in ["BN", "GN", "LN"]

        if norm_type == "BN":
            self.norm = nn.BatchNorm1d(num_features=norm_dim, **kwargs)
        elif norm_type == "GN":
            self.norm = nn.GroupNorm(num_channels=norm_dim, **kwargs)
        elif norm_type == "LN":
            self.norm = nn.Sequential(
                Rearrange("b c t -> b t c"),
                nn.LayerNorm(norm_dim),
                Rearrange("b t c-> b c t"),
            )

    def forward(self, x):
        return self.norm(x)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)

    def forward(self, x):
        assert x.dim() == 3
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)

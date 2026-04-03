import torch
import torch.nn as nn
from functools import partial
from typing import Union, List


class SpatialShift(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Since we can only shift the dimensions either left or right
        # We simply divide the features into two and shift left and right by the channel
        _, f, e = x.size()
        x[:, 1:, : e // 2] = x[
            :, : f - 1, : e // 2
        ]  # Shift the features to the left by 1
        x[:, : f - 1, e // 2 :] = x[
            :, 1:, e // 2 :
        ]  # Shift the features to the right by 1

        return x

class PolyMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_spatial=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.bias = bias
        self.use_spatial = use_spatial
        self.norm1 = norm_layer(self.hidden_features)
        self.norm3 = norm_layer(self.hidden_features)

        self.hidden_features = hidden_features
        self.U1 = nn.Linear(self.in_features, self.hidden_features, bias=bias)
        self.U2 = nn.Linear(self.in_features, self.hidden_features // 8, bias=bias)
        self.U3 = nn.Linear(self.hidden_features // 8, self.hidden_features, bias=bias)
        self.C = nn.Linear(self.hidden_features, self.out_features, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))

        if self.use_spatial:
            self.spatial_shift = SpatialShift()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.U1.weight)
        nn.init.kaiming_normal_(self.U2.weight)
        nn.init.kaiming_normal_(self.U3.weight)
        if self.bias:
            nn.init.ones_(self.U1.bias)
            nn.init.ones_(self.U2.bias)
            nn.init.ones_(self.U3.bias)

    def forward(self, x):  #
        if self.use_spatial:
            out1 = self.U1(x)
            out2 = self.U2(x)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out2 = self.U3(out2)
            out1 = self.norm1(out1)
            out2 = self.norm3(out2)
            out_so = out1 * out2

        else:
            out1 = self.U1(x)
            out2 = self.U2(x)
            out2 = self.U3(out2)
            out1 = self.norm1(out1)
            out2 = self.norm3(out2)
            out_so = out1 * out2

        out1 = (1.0 - self.alpha) * out1 + self.alpha * out_so
        del out_so

        out1 = self.C(out1)

        return out1
    

class PolyBlock(nn.Module):
    def __init__(
        self,
        in_features: Union[int, List[int]],
        out_features: Union[int, List[int]],
        embed_dim: Union[int, List[int]],
        expansion_factor: int = 3,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        spatial_mix: bool = False,
    ):
        super().__init__()

        self.expansion_factor = expansion_factor

        if isinstance(in_features, int):
            in_features = 2 * [in_features]
        if isinstance(embed_dim, int):
            embed_dim = 2 * [embed_dim]
        if isinstance(out_features, int):
            out_features = 2 * [out_features]

        self.norm1 = norm_layer(in_features[0])
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.alpha3 = nn.Parameter(torch.zeros(1))

        self.mlp1 = PolyMLP(
            in_features=in_features[0],
            hidden_features=embed_dim[0],
            out_features=out_features[0],
            use_spatial=spatial_mix,
            bias=False,
        )
        self.mlp2 = PolyMLP(
            in_features=in_features[1],
            hidden_features=embed_dim[1] * expansion_factor,
            out_features=out_features[1],
            use_spatial=False,
            bias=False,
        )

    def forward(self, x):
        x_skip = x
        z = self.norm1(x)
        z = self.mlp1(z)
        x = self.alpha1 * x + z
        z = self.norm1(x)
        z = self.mlp2(z)
        x = self.alpha2 * x + z
        return self.alpha3 * x_skip + x

    

if __name__ == "__main__":
    mixer = SpatialShift()
    a = torch.randn(1, 3, 3, 8)
    print("Initial Embedding:")
    print(a)

    print("After mixing: ")
    print(mixer(a))

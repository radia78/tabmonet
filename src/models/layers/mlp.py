import torch
import torch.nn as nn
from functools import partial


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


class FeatureFilter(nn.Module):
    def __init__(
        self, n_features: int, feature_dim: int, embed_dim: int, rank: int = 4
    ):
        super().__init__()
        self.emb_filter = nn.Linear(feature_dim, rank, bias=False)
        self.feature_filter = nn.Linear(n_features, rank, bias=False)
        self.out_layer = nn.Linear(feature_dim * n_features, embed_dim, bias=False)
        self.norm1 = nn.LayerNorm(rank)
        self.norm2 = nn.LayerNorm(rank)

    def forward(self, x):
        a = self.norm1(self.emb_filter(x))  # (batch size, n_features, rank)
        b = self.norm2(
            self.feature_filter(x.transpose(1, -1))
        )  # (batch size, feature_dim, rank)
        c = torch.einsum("bik, bjk -> bij", a, b)
        o = self.out_layer((c * x + x).flatten(1, -1))

        return o


class PolyMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        n_degree=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_alpha=True,
        use_spatial=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.bias = bias
        self.use_spatial = use_spatial
        self.norm1 = norm_layer(self.hidden_features)
        self.norm3 = norm_layer(self.hidden_features)

        self.n_degree = n_degree
        self.hidden_features = hidden_features
        self.U1 = nn.Linear(self.in_features, self.hidden_features, bias=bias)
        self.U2 = nn.Linear(self.in_features, self.hidden_features // 8, bias=bias)
        self.U3 = nn.Linear(self.hidden_features // 8, self.hidden_features, bias=bias)
        self.C = nn.Linear(self.hidden_features, self.out_features, bias=True)
        self.eta = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))

        if self.use_spatial:
            self.spatial_shift = SpatialShift()
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
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
            out_so = (self.eta * out1) * (self.beta * out2)

        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            del out_so

        else:
            out1 = out1 + out_so
            del out_so

        out1 = self.C(out1)

        return out1
    

if __name__ == "__main__":
    mixer = SpatialShift()
    a = torch.randn(1, 3, 3, 8)
    print("Initial Embedding:")
    print(a)

    print("After mixing: ")
    print(mixer(a))

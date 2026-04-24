import torch
import torch.nn as nn
from typing import Optional
from functools import partial


class NewtonRaphsonLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, iterations: int = 3):
        super().__init__()
        self.eps = eps
        self.iterations = iterations
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        v = var + self.eps
        y = 1.0 / torch.clamp(v, min=self.eps)

        for _ in range(self.iterations):
            y = 0.5 * y * (3.0 - v * (y**2))

        x_hat = (x - mean) * y
        return self.gamma * x_hat + self.beta


# TODO:
# - Current implementation uses a for-loop which is slow but allows each slice of the tensor be updated individual
# - This will be a unique optimization challenge since a single parameter will be hard to optimize since the derivative will be a higher order tensor
class EnsembleAdapter(nn.Module):
    def __init__(self, n_estimator: int, in_features: int, out_features: int):
        super().__init__()
        modules = []
        for _ in range(n_estimator):
            modules.append(nn.Linear(in_features, out_features, bias=False))
        self.estimators = nn.ModuleList(modules)

    def forward(self, x):
        estimates = torch.stack(
            [estimator(x.flatten(1, -1)) for estimator in self.estimators], dim=1
        )
        return estimates


class ConvolutionEnsemble(nn.Module):
    def __init__(self, n_features: int, feature_dim: int, out_dim: int):
        super().__init__()
        self.padding = nn.CircularPad1d((0, 2))
        self.upsize = nn.Linear(feature_dim, int(4 / 3 * out_dim), bias=False)
        self.ensemble = nn.Conv1d(
            in_channels=int(4 / 3 * out_dim),
            out_channels=out_dim,
            bias=False,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.upsize(x)
        x = self.padding(x.transpose(1, -1))
        x = self.ensemble(x)

        return x.transpose(1, -1)


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
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer_norm: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        bias: bool = True,
        use_spatial: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.bias = bias
        self.use_spatial = use_spatial

        self.norm1 = layer_norm(self.hidden_features)
        self.norm3 = layer_norm(self.hidden_features)

        self.U1 = nn.Linear(self.in_features, self.hidden_features, bias=self.bias)
        self.U2 = nn.Linear(self.in_features, self.hidden_features // 8, bias=self.bias)
        self.U3 = nn.Linear(
            self.hidden_features // 8, self.hidden_features, bias=self.bias
        )
        self.C = nn.Linear(self.hidden_features, self.out_features, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))

        if self.use_spatial:
            self.spatial_shift = SpatialShift()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.U1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.U2.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.U3.weight, nonlinearity="linear")
        if self.bias:
            nn.init.ones_(self.U1.bias)
            nn.init.ones_(self.U2.bias)
            nn.init.ones_(self.U3.bias)

    def forward(self, x):
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
        hidden_dim: int,
        expansion_factor: int = 3,
        layer_norm: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        spatial_mix: bool = False,
    ):
        super().__init__()
        self.expansion_factor = expansion_factor
        self.hidden_dim = hidden_dim

        self.norm = layer_norm(hidden_dim, eps=1e-6)
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.alpha3 = nn.Parameter(torch.zeros(1))
        self.mlp1 = PolyMLP(
            layer_norm=layer_norm,
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            use_spatial=spatial_mix,
            bias=False,
        )
        self.mlp2 = PolyMLP(
            layer_norm=layer_norm,
            in_features=hidden_dim,
            hidden_features=hidden_dim * expansion_factor,
            out_features=hidden_dim,
            use_spatial=False,
            bias=False,
        )

    def forward(self, x):
        x_skip = x
        z = self.norm(x)
        z = self.mlp1(z)
        x = self.alpha1 * x + z
        z = self.norm(x)
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

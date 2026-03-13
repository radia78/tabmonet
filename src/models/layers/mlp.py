import torch
import torch.nn as nn
from functools import partial


class PolyMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_alpha=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.bias = bias
        self.norm1 = norm_layer(self.hidden_features)
        self.norm3 = norm_layer(self.hidden_features)

        self.hidden_features = hidden_features
        self.U1 = nn.Linear(self.in_features, self.hidden_features, bias=bias)
        self.U2 = nn.Linear(self.in_features, self.hidden_features // 8, bias=bias)
        self.U3 = nn.Linear(self.hidden_features // 8, self.hidden_features, bias=bias)
        self.C = nn.Linear(self.hidden_features, self.out_features, bias=True)

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

    def forward(self, x):
        out1 = self.U1(x)
        out2 = self.U2(x)
        out2 = self.U3(out2)
        out1 = self.norm1(out1)
        out2 = self.norm3(out2)
        out_so = out1 * out2

        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            del out_so

        else:
            out1 = out1 + out_so
            del out_so

        out1 = self.C(out1)

        return out1


class MultiHeadPolyMLP(nn.Module):
    def __init__(
        self,
        n_estimator: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        use_alpha: bool = False,
    ):
        super().__init__()
        self.n_estimator = n_estimator
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.norm1 = norm_layer(self.hidden_features)
        self.norm3 = norm_layer(self.hidden_features)

        self.hidden_features = hidden_features
        # We instatiate n_estimator MLPs as heads but keep the low-rank mlp only one
        self.U1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features, self.hidden_features, bias=bias)
                )
                for _ in range(n_estimator)
            ]
        )
        self.U2 = nn.Linear(self.in_features, self.hidden_features // 8, bias=bias)
        self.U3 = nn.Linear(self.hidden_features // 8, self.hidden_features, bias=bias)
        self.C = nn.Linear(
            self.hidden_features * self.n_estimator, self.out_features, bias=True
        )

        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))

        self.init_weights()

    def init_weights(self):
        # Initialize the multihead component
        for i in range(self.n_estimator):
            nn.init.kaiming_normal_(self.U1[i].weight)
            nn.init.ones_(self.U1[i].bias)

        nn.init.kaiming_normal_(self.U2.weight)
        nn.init.kaiming_normal_(self.U3.weight)
        nn.init.ones_(self.U2.bias)
        nn.init.ones_(self.U3.bias)

    def forward(self, x):
        out1_list = []
        for i in range(self.n_estimator):
            out1_list.append(self.U1[i](x))
        out1 = torch.stack(out1_list, dim=1)

        out2 = self.U2(x)
        out2 = self.U3(out2)
        out1 = self.norm1(out1)
        out2 = self.norm3(out2)

        out_so = out1 * out2.unsqueeze(1)

        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            del out_so

        else:
            out1 = out1 + out_so
            del out_so

        out1 = self.C(out1.flatten(1, -1))

        return out1

    def forward(self, x):
        # Input shape: (batch size, num_features, in_features)
        feature_mix = self.feature_poly_block(x)  # Shape: (batch_size, num_features, r)
        embedding_mix = self.embedding_poly_block(
            x.transpose(1, -1)
        )  # Shape: (batch_size, in_features, r)
        filtered_input = torch.einsum("bjk, bik -> bji", feature_mix, embedding_mix)

        return self.norm(filtered_input) + x


@DeprecationWarning
class KFilter(nn.Module):
    def __init__(self, r: int, num_features: int, in_features: int):
        super().__init__()
        self.quantile_filter = nn.Linear(in_features, r, bias=False)
        self.feature_filter = nn.Linear(num_features, r, bias=False)
        self.norm = nn.LayerNorm(in_features)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.quantile_filter.weight)
        nn.init.kaiming_normal_(self.feature_filter.weight)

    def forward(self, x):
        # Shape of input: (batch_size, num_features, embed_dim)
        filtered_feature = self.feature_filter(
            x.transpose(1, -1)
        )  # (batch_size, embed_dim, r)
        filtered_quantile = self.quantile_filter(x)  # (batch_size, num_features, r)

        filtered_input = torch.einsum(
            "bjk, bik -> bji", filtered_quantile, filtered_feature
        )

        return (self.norm(filtered_input) + x).flatten(
            1, -1
        )  # (batch_size, num_features, embed_dim)

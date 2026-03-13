import numpy as np
import torch
import torch.nn as nn


class CategoricalEmbedding(nn.Module):
    def __init__(self, num_features: int, max_class: int, embedding_size: int):
        super().__init__()

        self.num_features = num_features

        self.register_buffer(
            "embedding",
            nn.Parameter(torch.empty(num_features, max_class, embedding_size)),
        )
        nn.init.normal_(
            self.embedding
        )  # Purely heuristic, I have no idea what initialization scheme works for embeddings -> Need to learn more

    def forward(self, x):
        feature_idx = torch.arange(self.num_features, dtype=torch.long, device=x.device)
        emb = self.embedding[feature_idx, x, :]

        return emb


class LinearEmbedding(nn.Module):
    def __init__(self, num_features: int, emb_dim: int, bias: bool):
        super().__init__()
        self.bias = bias
        self.emb_layer = nn.Parameter(torch.empty(num_features, emb_dim))
        nn.init.uniform_(self.emb_layer, a=-np.pi, b=np.pi)

        if self.bias:
            self.bias_layer = nn.Parameter(torch.empty(num_features, emb_dim))
            nn.init.uniform_(self.bias_layer, a=-np.pi, b=np.pi)

    def forward(self, x):
        out = torch.einsum("ij, bi -> bij", self.emb_layer, x)
        if self.bias:
            out = out + self.bias_layer

        return out


class QuantileEmbeddingNoFraction(nn.Module):
    def __init__(
        self,
        num_bins: int,
        bin_edges: torch.Tensor,
        with_linear: bool,
        linear_emb_dim: int,
        bias: bool,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.with_linear = with_linear
        self.eps = eps
        self.register_buffer("bin_edges", bin_edges.T)
        self.num_bins = num_bins

        if with_linear:
            self.l = nn.Linear(self.num_bins, linear_emb_dim, bias=bias)

    def forward(self, x):
        batch_size, features = x.shape

        edges_upper = self.bin_edges[:, 1:]

        encoded = torch.zeros(batch_size, features, self.num_bins, device=x.device)
        encoded = torch.where(x.unsqueeze(-1) > edges_upper, 1.0, encoded)

        if self.with_linear:
            encoded = self.l(encoded)

        return encoded


class QuantileEmbedding(nn.Module):
    def __init__(
        self,
        num_bins: int,
        bin_edges: torch.Tensor,
        with_linear: bool,
        linear_emb_dim: int,
        bias: bool,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.with_linear = with_linear
        self.eps = eps
        self.register_buffer("bin_edges", bin_edges)
        self.num_bins = num_bins

        if with_linear:
            self.l = nn.Linear(self.num_bins, linear_emb_dim, bias=bias)

    def forward(self, x):
        batch_size, features = x.shape

        edges_upper = self.bin_edges[:, 1:]
        edges_lower = self.bin_edges[:, :-1]

        encoded = torch.zeros(batch_size, features, self.num_bins, device=x.device)
        encoded = torch.where(x.unsqueeze(-1) > edges_upper, 1.0, encoded)

        frac = (x.unsqueeze(-1) - edges_lower).div(edges_upper - edges_lower + self.eps)
        mask = (x.unsqueeze(-1) >= edges_lower) & (x.unsqueeze(-1) <= edges_upper)
        encoded = torch.where(mask, frac, encoded)

        if self.with_linear:
            encoded = self.l(encoded)

        return encoded


class QuantileEmbeddingV2(nn.Module):
    def __init__(
        self, num_bins: int, bin_edges: torch.Tensor, bias: bool, eps: float = 1e-5
    ):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.register_buffer("bin_edges", bin_edges)
        self.num_bins = num_bins
        num_features = self.bin_edges.shape[0]

        if bias:
            self.offset = nn.Parameter(torch.randn(num_features, num_bins))

    def forward(self, x):
        batch_size, features = x.shape

        edges_upper = self.bin_edges[:, 1:]
        edges_lower = self.bin_edges[:, :-1]

        encoded = torch.zeros(batch_size, features, self.num_bins, device=x.device)
        encoded = torch.where(x.unsqueeze(-1) > edges_upper, 1.0, encoded)

        frac = (x.unsqueeze(-1) - edges_lower).div(edges_upper - edges_lower + self.eps)
        mask = (x.unsqueeze(-1) >= edges_lower) & (x.unsqueeze(-1) <= edges_upper)
        encoded = torch.where(mask, frac, encoded) * x.unsqueeze(-1)

        if self.bias:
            encoded = encoded + self.offset

        return encoded


class PeriodicEmbedding(nn.Module):
    def __init__(
        self,
        num_features: int,
        sigma: float,
        emb_dim: int,
        with_linear: bool,
        linear_emb_dim: int,
        bias: bool,
    ):
        super().__init__()
        self.sigma = sigma
        self.with_linear = with_linear
        self.c = nn.Parameter(torch.empty(num_features, emb_dim))

        if with_linear:
            self.l = nn.Linear(2 * emb_dim, linear_emb_dim, bias=bias)

    def init_weights_(self):
        nn.init.normal_(self.c, std=self.sigma)

    def forward(self, x):
        out1 = torch.einsum("ij, bi -> bij", self.c, x)

        if self.with_linear:
            out2 = self.l(torch.cat((torch.cos(out1), torch.sin(out1)), dim=-1))

        else:
            out2 = torch.cat((torch.cos(out1), torch.sin(out1)), dim=-1)

        return out2


class PBLDEmbedding(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_features, 16))
        self.b1 = nn.Parameter(torch.empty(num_features, 16))
        self.w2 = nn.Linear(in_features=16, out_features=3, bias=True)

        self.init_weights_()

    def init_weights_(self):
        nn.init.normal_(self.w1, std=0.1)
        nn.init.uniform_(self.b1, a=-np.pi, b=np.pi)

    def forward(self, x):
        out1 = torch.einsum("ij, bi -> bij", self.w1, x)
        out2 = torch.cos(2 * torch.pi * out1 + self.b1)
        out3 = torch.cat((x.unsqueeze(-1), self.w2(out2)), dim=-1)

        return out3

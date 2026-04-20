import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    """
    Base embedding class for both categorical and numerical embedding

    Args:
        n_features (int): Number of categorical/numerical features
        emb_dim (int): The dimension of the embedding
    """

    def __init__(self, n_features: int, emb_dim: int):
        super().__init__()
        self.n_features = n_features
        self.emb_dim = emb_dim


class LinearEmbedding(Embedding):
    def __init__(self, n_features: int, emb_dim: int, bias: bool):
        """
        Linear embedding for numerical features based on http://arxiv.org/abs/2203.05556


        Args:
            n_features (int): Number of categorical/numerical features
            emb_dim (int): The dimension of the embedding
            bias (bool): Add a bias after multiplying the weights
        """
        super().__init__(n_features, emb_dim)
        self.bias = bias
        self.W = nn.Parameter(torch.empty(self.n_features, self.emb_dim))
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.n_features, self.emb_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.W, a=-math.pi, b=math.pi)
        if self.bias:
            nn.init.uniform_(self.b, a=-math.pi, b=math.pi)

    def forward(self, x):
        x = self.W * x.unsqueeze(-1)
        if self.bias:
            x = x + self.b

        return x


class PeriodicEmbedding(Embedding):
    def __init__(
        self,
        n_features: int,
        emb_dim: int,
        sigma: float,
        bias: bool,
    ):
        """
        Periodic embeddings specified by http://arxiv.org/abs/2203.05556

        Args:
            n_features (int): Number of categorical/numerical features
            hidden_dim (int): The intermediate dimension for the first weight C
            sigma (int): The variance of the initial weight values of C
            bias (bool): Add bias for the linear layer W
        """
        super().__init__(n_features, emb_dim)
        self.sigma = sigma
        self.bias = bias
        self.C = nn.Parameter(torch.empty(self.n_features, self.emb_dim // 2))

    def forward(self, x):
        x = 2 * math.pi * self.C * x.unsqueeze(-1)
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)

        return x


class PBLDEmbedding(Embedding):
    def __init__(self, n_features: int, emb_dim: int = 3, hidden_dim: int = 16):
        super().__init__(n_features, emb_dim)
        self.W1 = nn.Parameter(torch.empty(n_features, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(n_features, hidden_dim))
        self.W2 = nn.Linear(hidden_dim, emb_dim, bias=True)

        self._init_weights_()

    def _init_weights_(self):
        nn.init.normal_(self.W1, std=0.1)
        nn.init.uniform_(self.b1, a=-math.pi, b=math.pi)

    def forward(self, x):
        e = 2 * math.pi * self.W1 * x.unsqueeze(-1)
        e = e + self.b1
        e = torch.cos(e)
        e = torch.cat((x.unsqueeze(-1), self.W2(e)), dim=-1)

        return e


class QuantileEmbedding(Embedding):
    def __init__(
        self, n_features: int, emb_dim: int, bin_edges: torch.Tensor, eps: float = 1e-8
    ):
        super().__init__(n_features, emb_dim)
        self.eps = eps
        assert (
            bin_edges.shape[0] == self.n_features
            and bin_edges.shape[1] == self.emb_dim + 1
        ), "Quantile Bin Tensors must have shape (n_features, emb_dim + 1)"
        self.register_buffer("bin_edges", bin_edges)

    def forward(self, x):
        edges_upper = self.bin_edges[:, 1:]
        edges_lower = self.bin_edges[:, :-1]

        encoded = torch.zeros(
            x.shape[0], self.n_features, self.emb_dim, device=x.device
        )
        encoded = torch.where(x.unsqueeze(-1) > edges_upper, 1.0, encoded)

        frac = (x.unsqueeze(-1) - edges_lower).div(edges_upper - edges_lower + self.eps)
        mask = (x.unsqueeze(-1) >= edges_lower) & (x.unsqueeze(-1) <= edges_upper)
        encoded = torch.where(mask, frac, encoded)

        return encoded


class CategoricalEmbedding(nn.Module):
    def __init__(self, cardinalities: list[int], emb_dim: int):
        super().__init__()
        self.cardinalities = cardinalities
        embeddings = []
        for k in cardinalities:
            embeddings.append(nn.Embedding(k + 1, emb_dim, padding_idx=0))
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x > 0) * x
        temp = []
        for i in range(len(self.cardinalities)):
            temp.append(self.embeddings[i](x[:, i]))
        encoding = torch.stack(temp, dim=1)
        return encoding

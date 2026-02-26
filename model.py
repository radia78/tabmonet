import torch
import torch.nn as nn
from functools import partial

class PolyMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            bias=True,
            drop=0.5,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_alpha=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.norm1 = norm_layer(self.hidden_features)
        self.norm3 = norm_layer(self.hidden_features)

        self.hidden_features = hidden_features
        self.U1 = nn.Linear(self.in_features, self.hidden_features, bias=bias)
        self.U2 = nn.Linear(self.in_features, self.hidden_features//8, bias=bias)
        self.U3 = nn.Linear(self.hidden_features//8, self.hidden_features, bias=bias)
        self.C = nn.Linear(self.hidden_features, self.out_features, bias=True) 
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

        if self.use_alpha:
              self.alpha = nn.Parameter(torch.ones(1))
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.U1.weight)
        nn.init.kaiming_normal_(self.U2.weight)
        nn.init.kaiming_normal_(self.U3.weight)
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
    
class TabularMONet(nn.Module):
    def __init__(
            self,
            num_features, 
            embed_dim,
            expansion_factor,
            num_embedding,
            num_layers,
            task='reg',
            num_classes=10,
        ):
        super().__init__()
        modules = []
        for i in range(num_layers):
            in_features = num_features if i == 0 else embed_dim
            modules.append(
                nn.Sequential(
                    PolyMlp(in_features, embed_dim, embed_dim),
                    PolyMlp(embed_dim, (embed_dim) * expansion_factor, embed_dim)
                )
            )
        
        self.num_embedding = num_embedding

        self.poly_blocks = nn.Sequential(*modules)
        if task == 'reg':
            self.head = nn.Linear(embed_dim, out_features=1)
        else:
            self.head = nn.Linear(embed_dim, out_features=num_classes)

        self.init_head_weights_()

    def init_head_weights_(self):
        nn.init.kaiming_normal_(self.head.weight)
        nn.init.ones_(self.head.bias)
    
    def forward(self, x_num, x_cat):
        if x_cat is not None and x_num is None:
            e = x_cat

        if x_num is not None and x_cat is None:
            e = self.num_embedding(x_num).flatten(start_dim=1)

        else:
            e = torch.cat([
                self.num_embedding(x_num).flatten(start_dim=1),
                x_cat,
            ], dim=-1)

        h = self.poly_blocks(e)
        o = self.head(h)
        
        return o
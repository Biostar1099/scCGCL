
import os
import gc
import time
import psutil
import numpy as np
import pandas as pd
import anndata
import scanpy as sc


from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_undirected, add_self_loops

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_undirected, add_self_loops


class GATEncoder(nn.Module):
    def __init__(self, num_genes, latent_dim, num_heads=10, dropout=0.2):
        super().__init__()
        self.gat_layer_1 = GATConv(
            in_channels=num_genes, out_channels=128, heads=num_heads, dropout=dropout, concat=True
        )
        in_dim2 = 128 * num_heads
        self.gat_layer_2 = GATConv(
            in_channels=in_dim2, out_channels=latent_dim, heads=num_heads, concat=False
        )

    def forward(self, x, edge_index):
        h = F.relu(self.gat_layer_1(x, edge_index))
        z = F.relu(self.gat_layer_2(h, edge_index))
        return z




class GCN(nn.Module):
    def __init__(self, num_features, output_dim):
        super().__init__()
        self.conv1 = GCNConv(num_features, output_dim, bias=True)
        self.prelu = nn.PReLU(output_dim)
        nn.init.xavier_normal_(self.conv1.lin.weight)
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x, edge_index):
        z = self.prelu(self.conv1(x, edge_index))
        return z


def loss_infomax_new(x, x_cl, edge_index, T: float = 0.1):
    
    N = x.size(0)
    device = x.device
    x_norm = F.normalize(x, dim=1)
    x_cl_norm = F.normalize(x_cl, dim=1)
    sim = torch.exp(torch.mm(x_norm, x_cl_norm.t()) / T)  # [N,N]
    sim_1=torch.exp(torch.mm(x_norm, x_norm.t()) / T)
    pos = sim.diag()

    mask = torch.ones_like(sim, dtype=torch.bool, device=device)
    idx = torch.arange(N, device=device)
    mask[idx, idx] = False
    if edge_index.numel() > 0:
        src, dst = edge_index
        mask[src, dst] = False
        mask[dst, src] = False
    mask_1=torch.ones_like(sim_1, dtype=torch.bool, device=device)
    idx_1 = torch.arange(N, device=device)
    mask_1[idx_1, idx_1] = False
    if edge_index.numel() > 0:
        src, dst = edge_index
        mask_1[src, dst] = False
        mask_1[dst, src] = False
    neg_sum = (sim * mask.float()).sum(dim=1)
    neg_sum_1=(sim_1 * mask_1.float()).sum(dim=1)
    loss = -torch.log(pos / (pos + neg_sum+neg_sum_1)).mean()
    return loss

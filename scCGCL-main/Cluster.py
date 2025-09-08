import os
import gc
import time
import psutil
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_undirected, add_self_loops



def correlation(data_numpy, k, corr_type='pearson'):
    """基于基因表达相关性（行=细胞）构建 KNN（返回邻接对）。"""
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    return corr, neighbors

def compute_metrics(y_true, y_pred):
    return {"ARI": ARI(y_true, y_pred), "NMI": NMI(y_true, y_pred)}

def cluster(embedding, gt_labels, num_cluster, cluster_name='kmeans', seed=42):
    # 保持与你的接口一致
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
    imputer = SimpleImputer(strategy='mean')
    embedding = imputer.fit_transform(embedding)

    if cluster_name == 'kmeans':
        pd_labels = KMeans(n_clusters=num_cluster, random_state=seed).fit(embedding).labels_
        eval_supervised_metrics = compute_metrics(gt_labels, pd_labels)
        return eval_supervised_metrics, pd_labels
    else:
        raise NotImplementedError("只保留了 kmeans 以精简依赖。")


# ============================ 谱聚类 + Top-N 选择 ============================

def _spectral_embed_and_labels(adj_csr: sp.csr_matrix, k: int):
    """谱嵌入 + KMeans 标签。返回 (X, labels)。
    X: [num_nodes, k] 前 k 个特征向量。
    labels: [num_nodes] 谱聚类标签。
    """
    norm_laplacian = sp.csgraph.laplacian(adj_csr, normed=True)
    u, s, _ = svds(norm_laplacian, k=k)
    eigvecs = u[:, ::-1]  # 按从小到大顺序
    X = torch.tensor(eigvecs[:, :k], dtype=torch.float32)
    X_scaled = StandardScaler().fit_transform(X.numpy())
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def _centrality_scores(X: torch.Tensor, labels: torch.Tensor):
    """对每个簇，给出该簇内每个节点的范数分数列表（与原 cal_centrality 逻辑一致）。"""
    scores = []
    for label in torch.unique(labels):
        mask = (labels == label)
        if mask.any():
            scores.append(torch.norm(X[mask], dim=1))
        else:
            scores.append(torch.tensor([], dtype=torch.float32))
    return scores  # list[Tensor]


def select_topN_per_cluster(adj_csr: sp.csr_matrix,
                            k: int,
                            feature_matrix,  # ndarray / csr / torch.Tensor
                            N: int):
    """选择每簇 Top-N 谱中心性细胞。

    返回：
      core_indices_all: 1D Tensor (长度 N*k) —— 选中的核心细胞（全局索引）
      core_indices_per_cluster: List[1D Tensor] —— 每个簇的核心细胞（全局索引）
      core_features: Tensor 形状 [N*k, genes]
    """
    # 统一为 Tensor
    if not isinstance(feature_matrix, torch.Tensor):
        feature_matrix = torch.tensor(
            feature_matrix.toarray() if hasattr(feature_matrix, 'toarray') else feature_matrix,
            dtype=torch.float32
        )

    X_spec, labels = _spectral_embed_and_labels(adj_csr, k)
    cent_scores = _centrality_scores(X_spec, labels)

    core_indices_per_cluster = []
    unique_labels = torch.unique(labels)
    for lab in unique_labels:
        mask = (labels == lab)
        idx_in_cluster = torch.where(mask)[0]
        if len(idx_in_cluster) == 0:
            core_indices_per_cluster.append(torch.tensor([], dtype=torch.long))
            continue
        scores = cent_scores[lab.item()]
        order = torch.argsort(scores, descending=True)
        topN_local = order[:min(N, len(order))]
        core_idx_global = idx_in_cluster[topN_local]
        core_indices_per_cluster.append(core_idx_global)

    if len(core_indices_per_cluster) > 0:
        core_indices_all = torch.cat(core_indices_per_cluster)
        core_features = feature_matrix[core_indices_all]
    else:
        core_indices_all = torch.tensor([], dtype=torch.long)
        core_features = torch.tensor([], dtype=torch.float32)

    return core_indices_all, core_indices_per_cluster, core_features


# ============================ 子图与边构造（**新逻辑**） ============================

def complete_graph_edges_for_clusters(core_indices_per_cluster):
    """为每个簇的核心节点构建 **完全图** 边（全局索引对）。"""
    edges = []
    for core_idx in core_indices_per_cluster:
        arr = core_idx.cpu().numpy().tolist()
        L = len(arr)
        for i in range(L):
            u = arr[i]
            for j in range(i + 1, L):
                v = arr[j]
                edges.append((u, v))
                edges.append((v, u))
    return edges


def edges_touching_core_from_global(global_edge_index: torch.Tensor,
                                    core_set: set):
    """从 global_graph 中提取与核心节点相接的 1 阶邻居边（保留方向对）。
    仅保留 (u,v) 中至少一端属于 core_set 的边。
    返回：edges(list[Tuple[int,int]]), neighbors_only(set[int])
    """
    src = global_edge_index[0].cpu().numpy()
    dst = global_edge_index[1].cpu().numpy()
    edges = []
    neighbors = set()
    for u, v in zip(src, dst):
        if (u in core_set) or (v in core_set):
            edges.append((int(u), int(v)))
            # 记录非核心的一端作为邻居
            if u in core_set and v not in core_set:
                neighbors.add(int(v))
            if v in core_set and u not in core_set:
                neighbors.add(int(u))
    return edges, neighbors


def reindex_subgraph(core_indices_all: torch.Tensor,
                     neighbor_indices_set: set,
                     edges_global_pairs):
    """将 (核心 + 邻居) 节点重编号为 [0..M+R-1]，核心节点排在最前。
    返回：
      new_edge_index [2,E],
      new_node_order List[int] (长度 M+R; 前 M 个就是核心的新顺序),
      core_new_index Tensor (长度 M),
      mapping 原->新（dict）
    """
    core_list = core_indices_all.cpu().numpy().tolist()
    neighbors_list = sorted(list(neighbor_indices_set - set(core_list)))

    new_node_order = core_list + neighbors_list
    mapping = {old: new for new, old in enumerate(new_node_order)}

    # 重新映射边
    mapped = []
    for u, v in edges_global_pairs:
        if (u in mapping) and (v in mapping):
            mapped.append((mapping[u], mapping[v]))
    if len(mapped) == 0:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        new_edge_index = torch.tensor(mapped, dtype=torch.long).t().contiguous()

    # 无向 + 自环（自环对 GAT/GCN 通常不必须，这里保持 add_self_loops 接口可选）
    new_edge_index = to_undirected(new_edge_index)
    new_edge_index, _ = add_self_loops(new_edge_index)

    core_new_index = torch.arange(len(core_list), dtype=torch.long)
    return new_edge_index, new_node_order, core_new_index, mapping


def build_core_loss_edges(core_indices_per_cluster, mapping):
    """仅为核心节点构建用于 loss 的完全图边（以新索引返回）。"""
    loss_pairs = []
    for core_idx in core_indices_per_cluster:
        arr = [mapping[int(x)] for x in core_idx.cpu().numpy().tolist() if int(x) in mapping]
        L = len(arr)
        for i in range(L):
            u = arr[i]
            for j in range(i + 1, L):
                v = arr[j]
                loss_pairs.append((u, v))
                loss_pairs.append((v, u))
    if len(loss_pairs) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(loss_pairs, dtype=torch.long).t().contiguous()

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
import torch.nn.functional as F


from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_undirected, add_self_loops


from layer import GATEncoder,loss_infomax_new
from perpare_graph import load_graph


from Cluster import cluster,select_topN_per_cluster,complete_graph_edges_for_clusters,edges_touching_core_from_global,reindex_subgraph,build_core_loss_edges




import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='prepare graph!')

    parser.add_argument('--graph_type', type=str, default='PKNN', choices=['KNN', 'PKNN'],
                        help='Type of graph to construct')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors for KNN graph')

    parser.add_argument('--input_h5ad_path', type=str, default="./data/deng_preprocessed.h5ad", help='path to input h5ad file')

    parser.add_argument('--save_graph_path', type=str, default="./graph/muraro.txt",help='path to save graph')

    parser.add_argument('--cluster_num', type=int, default=6, help='cluster_num')

    parser.add_argument('--latent_dim', type=int, default=256, help='latent_dim')

    parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')

    parser.add_argument('--epoch',type=int,default=600,help='epochs')


    return parser.parse_args()


def main():
    args = parse_arguments()
    # -------- 参数 --------
    h5ad_data_file = args.input_h5ad_path
    graph_txt = args.save_graph_path  # 若没有可用 prepare_graphs 生成

    args_graph_type = args.graph_type
    args_KNNs_K = args.k
    args_cluster_num = args.cluster_num

    args_latent_dim = args.latent_dim
    learning_rate = args.lr
    args_epochs = args.epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # -------- 读数据 & 全局图 --------
    processed_adata = sc.read_h5ad(h5ad_data_file)
    num_nodes, num_gene = processed_adata.shape
    processed_adata.X = processed_adata.X.astype(np.float32)

    # 读或建图
    if os.path.exists(graph_txt):
        edgelist = load_graph(graph_txt)
    #else:
        #edgelist = prepare_graphs(processed_adata, 'graph', './graph', args_graph_type, args_KNNs_K)

    edge_index_np = np.array(edgelist, dtype=int).T
    edge_index_global = to_undirected(torch.from_numpy(edge_index_np).long(), num_nodes)

    x_global = torch.tensor(processed_adata.X, dtype=torch.float32)
    global_graph = Data(x=x_global, edge_index=edge_index_global)

    # 稀疏邻接用于谱聚类
    adj_sp = coo_matrix(
        (np.ones(edge_index_global.shape[1]),
         (edge_index_global.cpu().numpy()[0], edge_index_global.cpu().numpy()[1])),
        shape=(num_nodes, num_nodes)
    ).tocsr()

    # -------- 谱聚类：每簇选 Top-N 核心节点（全局索引） --------
    core_indices_all, core_indices_per_cluster, core_features = select_topN_per_cluster(
        adj_sp, k=args_cluster_num, feature_matrix=processed_adata.X, N=10
    )
    assert len(core_indices_all) > 0, "未选出任何核心节点，检查参数 N/K 或谱分解。"

    # -------- 构建边：核心完全图 + 与核心相接的一阶邻居 --------
    core_edges_complete = complete_graph_edges_for_clusters(core_indices_per_cluster)
    core_set = set(core_indices_all.cpu().numpy().tolist())
    touch_edges, neighbor_set = edges_touching_core_from_global(edge_index_global, core_set)

    # 合并边（使用 set 去重）
    all_pairs = list({(int(u), int(v)) for (u, v) in (core_edges_complete + touch_edges)})

    # -------- 重编号：核心在前，邻居在后 --------
    sub_edge_index, new_node_order, core_new_index, mapping = reindex_subgraph(
        core_indices_all, neighbor_set, all_pairs
    )

    # 用于 loss 的核心边（完全图；仅核心节点的新索引）
    core_loss_edges = build_core_loss_edges(core_indices_per_cluster, mapping)

    # -------- 子图特征（按 new_node_order 提取），发送到设备 --------
    new_node_order_tensor = torch.tensor(new_node_order, dtype=torch.long)
    x_sub = x_global[new_node_order_tensor].to(device)
    sub_edge_index = sub_edge_index.to(device)
    core_loss_edges = core_loss_edges.to(device)
    core_new_index = core_new_index.to(device)

    # -------- 模型 --------
    model = GATEncoder(num_genes=num_gene, latent_dim=args_latent_dim).to(device)
    # model = GCN(num_features=num_gene, output_dim=args_latent_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    mem_before = process.memory_info().rss

    best_ari = 0.0
    best_nmi = 0.0

    for epoch in range(args_epochs):
        model.train()
        # 数据增强：仅对子图节点做（邻居用于消息传递，loss 只计算核心）
        noise = torch.normal(0, torch.ones_like(x_sub, device=device) * 0.01)
        aug = x_sub + F.normalize(noise, dim=1)

        z_aug = model(aug, sub_edge_index)
        z = model(x_sub, sub_edge_index)

        # 仅在 **核心节点** 上计算对比损失
        z_core = z[core_new_index]
        z_aug_core = z_aug[core_new_index]
        loss = 0.5 * (loss_infomax_new(z_core, z_aug_core, core_loss_edges) +
                      loss_infomax_new(z_aug_core, z_core, core_loss_edges))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                z_full = model(x_global.to(device), edge_index_global.to(device))
            eval_metrics, _ = cluster(z_full.detach().cpu(), processed_adata.obs['x'], num_cluster=args_cluster_num)
            ari, nmi = eval_metrics['ARI'], eval_metrics['NMI']
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f} | ARI={ari:.4f}, NMI={nmi:.4f}")
            if ari > best_ari:
                best_ari, best_nmi = ari, nmi

    elapsed = time.perf_counter() - start_time
    mem_after = process.memory_info().rss
    print('Best ARI, NMI:', best_ari, best_nmi)
    print(f"运行时间: {elapsed:.6f}秒")
    print(f"内存使用: {(mem_after - mem_before) / (1024 ** 2):.2f} MB")

    # 释放显存/内存
    del z, z_aug, z_core, z_aug_core
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()


if __name__ == '__main__':
    main()

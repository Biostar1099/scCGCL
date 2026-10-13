import os
import gc
import time
import psutil
import random
import argparse
from collections import defaultdict, Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

from layer import GATEncoder, loss_infomax_new
from prepare_graph import load_graph  # 修正拼写：perpare -> prepare

from Cluster import (
    cluster,
    select_topN_per_cluster,
    complete_graph_edges_for_clusters,
    edges_touching_core_from_global,
    reindex_subgraph,
    build_core_loss_edges,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_int_list(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(int).tolist()
    return [int(i) for i in x]


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_disjoint_core_neighbor_edges(edge_index_global, core_indices_per_cluster):
    """
    Build mutually exclusive cluster-specific subgraphs.

    Rule:
    1. Core cells are fixed to their own preliminary cluster.
    2. First-order neighbors are assigned to only one cluster.
       If a neighbor touches cores from multiple clusters, assign it to the cluster
       with the largest number of core-neighbor connections.
    3. Only edges within the same assigned cluster-specific subgraph are kept.
    """

    edge_np = edge_index_global.detach().cpu().numpy()
    num_nodes = int(edge_np.max()) + 1

    # Build adjacency list
    adj = [set() for _ in range(num_nodes)]
    for u, v in edge_np.T:
        u, v = int(u), int(v)
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    # Core ownership
    cluster_cores = [set(to_int_list(c)) for c in core_indices_per_cluster]

    owner = {}
    core_owner = {}

    for cid, cores in enumerate(cluster_cores):
        for c in cores:
            owner[c] = cid
            core_owner[c] = cid

    # Vote for first-order neighbors
    neighbor_votes = defaultdict(Counter)

    for cid, cores in enumerate(cluster_cores):
        for c in cores:
            for nb in adj[c]:
                if nb in core_owner:
                    continue
                neighbor_votes[nb][cid] += 1

    # Assign each neighbor to only one cluster
    for nb, counter in neighbor_votes.items():
        max_count = max(counter.values())
        candidate_clusters = [cid for cid, count in counter.items() if count == max_count]
        assigned_cid = min(candidate_clusters)
        owner[nb] = assigned_cid

    # Build mutually exclusive node sets
    cluster_node_sets = [set(cores) for cores in cluster_cores]
    neighbor_set = set()

    for node, cid in owner.items():
        if node not in core_owner:
            cluster_node_sets[cid].add(node)
            neighbor_set.add(node)

    # Build edges only inside each cluster-specific subgraph
    edge_set = set()

    # Complete graph among core cells within each cluster
    for cid, cores in enumerate(cluster_cores):
        cores = list(cores)
        for i in range(len(cores)):
            for j in range(len(cores)):
                if i != j:
                    edge_set.add((cores[i], cores[j]))

    # Keep only core-neighbor edges within the same assigned subgraph
    for cid, cores in enumerate(cluster_cores):
        nodes = cluster_node_sets[cid]

        for c in cores:
            for nb in adj[c]:
                if nb in nodes and owner.get(nb, None) == cid and nb != c:
                    edge_set.add((c, nb))
                    edge_set.add((nb, c))

    all_pairs = list(edge_set)
    # Safety check: node sets must be mutually exclusive
    for i in range(len(cluster_node_sets)):
        for j in range(i + 1, len(cluster_node_sets)):
            assert cluster_node_sets[i].isdisjoint(cluster_node_sets[j]), \
                f"Subgraph {i} and {j} are not disjoint."

    # Safety check: no cross-subgraph edges
    for u, v in all_pairs:
        assert owner[u] == owner[v], \
            f"Cross-subgraph edge detected: {u}-{v}, owner={owner[u]}-{owner[v]}"

    return all_pairs, neighbor_set, cluster_node_sets, owner


def parse_arguments():
    parser = argparse.ArgumentParser(description='scCGCL memory-friendly training')

    parser.add_argument('--graph_type', type=str, default='PKNN', choices=['KNN', 'PKNN'])
    parser.add_argument('--k', type=int, default=10)

    parser.add_argument('--input_h5ad_path', type=str, default="./data/deng_preprocessed.h5ad")
    parser.add_argument('--save_graph_path', type=str, default="./graph/deng.txt")

    parser.add_argument('--cluster_num', type=int, default=8)
    parser.add_argument('--core_num', type=int, default=10)

    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=600)

    parser.add_argument('--eval_device', type=str, default='cpu', choices=['cpu', 'cuda', 'none'],
                        help='Use cpu to avoid GPU OOM during full-graph evaluation.')

    parser.add_argument('--eval_cluster_method', type=str, default='kmeans',
                        choices=['kmeans', 'hdbscan'],
                        help='Clustering method used during evaluation.')

    parser.add_argument('--amp', action='store_true',
                        help='Use mixed precision training on CUDA to reduce GPU memory.')

    parser.add_argument('--seed', type=int, default=42)

    # ========== 基于loss的早停参数 ==========
    parser.add_argument('--patience_loss', type=int, default=10,
                        help='早停等待轮数：连续多少轮loss下降不足则触发早停')
    parser.add_argument('--min_delta_loss', type=float, default=0.05,
                        help='loss下降阈值，低于该值视为loss已经稳定，计数+1')
    # ========== ReduceLROnPlateau 参数 ==========
    parser.add_argument('--lr_factor', type=float, default=0.5, help='学习率衰减系数')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='scheduler等待轮数，loss不下降则降lr')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='学习率下限')


    return parser.parse_args()


def evaluate_full_graph(
    model,
    x_global_cpu,
    edge_index_global_cpu,
    labels,
    num_cluster,
    num_gene,
    latent_dim,
    eval_device='cpu',
    eval_cluster_method='kmeans',
    seed=42,
    return_embedding=False,
):
    """
    Full graph evaluation.
    Default: CPU evaluation to avoid GPU OOM on large datasets.
    eval_cluster_method can be 'kmeans' or 'hdbscan'.
    """

    model.eval()

    if eval_device == 'none':
        return None, None, None

    with torch.inference_mode():
        if eval_device == 'cpu':
            # Build a CPU model and load current parameters to avoid full graph GPU inference.
            eval_model = GATEncoder(num_genes=num_gene, latent_dim=latent_dim).cpu()
            state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            eval_model.load_state_dict(state_dict_cpu)
            eval_model.eval()

            z_full = eval_model(x_global_cpu, edge_index_global_cpu)

            del eval_model
            del state_dict_cpu

        else:
            # GPU evaluation, only use this if GPU memory is enough.
            device = torch.device('cuda')
            x_global_cuda = x_global_cpu.to(device)
            edge_index_global_cuda = edge_index_global_cpu.to(device)

            z_full = model(x_global_cuda, edge_index_global_cuda)

            del x_global_cuda
            del edge_index_global_cuda

        z_full_cpu = z_full.detach().cpu()
        del z_full

    clear_cuda()

    eval_metrics, _ = cluster(
        z_full_cpu,
        labels,
        num_cluster=num_cluster,
        cluster_name=eval_cluster_method,
        seed=seed,
    )

    ari = eval_metrics['ARI']
    nmi = eval_metrics['NMI']

    emb_np = z_full_cpu.numpy() if return_embedding else None
    del z_full_cpu
    clear_cuda()

    return ari, nmi, emb_np


def main():
    args = parse_arguments()
    set_seed(args.seed)

    h5ad_data_file = args.input_h5ad_path
    graph_txt = args.save_graph_path

    args_latent_dim = args.latent_dim
    learning_rate = args.lr
    args_epochs = args.epoch
    core_num = args.core_num

    train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.eval_device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. eval_device is changed to cpu.")
        args.eval_device = 'cpu'

    print('Training device:', train_device)
    print('Evaluation device:', args.eval_device)
    print('Evaluation clustering method:', args.eval_cluster_method)
    print('Core cell number per cluster:', core_num)
    print('Random seed:', args.seed)
    print('Early stop: loss-based | patience_loss:', args.patience_loss,
          '| min_delta_loss:', args.min_delta_loss)

    processed_adata = sc.read_h5ad(h5ad_data_file)
    print(type(processed_adata))
    # 查看obs全部字段名
    print(processed_adata.obs.columns.tolist())
    # 完整预览前10行观测
    print(processed_adata.obs.head(10))

    args_cluster_num = len(np.unique(processed_adata.obs['x']))
    num_nodes, num_gene = processed_adata.shape

    # Convert expression matrix to float32 dense matrix if needed.
    if sp.issparse(processed_adata.X):
        X_np = processed_adata.X.toarray().astype(np.float32)
    else:
        X_np = np.asarray(processed_adata.X, dtype=np.float32)

    if not os.path.exists(graph_txt):
        raise FileNotFoundError(f"Graph file not found: {graph_txt}")

    edgelist = load_graph(graph_txt)

    edge_index_np = np.array(edgelist, dtype=int).T
    edge_index_global_cpu = to_undirected(
        torch.from_numpy(edge_index_np).long(),
        num_nodes,
    ).cpu()

    x_global_cpu = torch.tensor(X_np, dtype=torch.float32).cpu()

    adj_sp = coo_matrix(
        (
            np.ones(edge_index_global_cpu.shape[1]),
            (
                edge_index_global_cpu.numpy()[0],
                edge_index_global_cpu.numpy()[1],
            ),
        ),
        shape=(num_nodes, num_nodes),
    ).tocsr()

    core_indices_all, core_indices_per_cluster, core_features = select_topN_per_cluster(
        adj_sp,
        k=args_cluster_num,
        feature_matrix=X_np,
        N=core_num,
    )
    assert len(core_indices_all) > 0

    print("Core cells per cluster:", [len(c) for c in core_indices_per_cluster])
    print("Total core cells:", len(core_indices_all))

    all_pairs, neighbor_set, cluster_node_sets, node_owner = build_disjoint_core_neighbor_edges(
        edge_index_global_cpu,
        core_indices_per_cluster,
    )

    print("Number of cluster-specific subgraphs:", len(cluster_node_sets))
    print("Subgraph sizes:", [len(s) for s in cluster_node_sets])
    print("Number of edges in disjoint reconstruction graph:", len(all_pairs))

    sub_edge_index, new_node_order, core_new_index, mapping = reindex_subgraph(
        core_indices_all,
        neighbor_set,
        all_pairs,
    )

    core_loss_edges = build_core_loss_edges(core_indices_per_cluster, mapping)

    # Only the subgraph used for training is moved to GPU.
    new_node_order_tensor = torch.tensor(new_node_order, dtype=torch.long)
    x_sub = x_global_cpu[new_node_order_tensor].to(train_device)
    sub_edge_index = sub_edge_index.to(train_device)
    core_loss_edges = core_loss_edges.to(train_device)
    core_new_index = core_new_index.to(train_device)

    model = GATEncoder(num_genes=num_gene, latent_dim=args_latent_dim).to(train_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # ReduceLROnPlateau：监控训练loss，mode=min
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.scheduler_patience,
        threshold=args.min_delta_loss,  # 旧版torch用 threshold，不是min_delta
        min_lr=args.min_lr,
        verbose=True
    )

    use_amp = args.amp and train_device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    mem_before = process.memory_info().rss

    best_ari = 0.0
    best_nmi = 0.0

    # ========== 基于loss的早停变量 ==========
    patience_loss_counter = 0
    patience_loss = args.patience_loss
    min_delta_loss = args.min_delta_loss
    prev_loss = float("inf")   # 上一轮loss，初始为无穷大
    saved_emb = False

    # 自动提取数据集名，用于npz自动命名
    dataset_name = os.path.basename(args.input_h5ad_path).replace(".h5ad", "")
    output_dir = "./emb_result"
    os.makedirs(output_dir, exist_ok=True)

    clear_cuda()

    for epoch in range(args_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Generate augmentation on GPU, then release it after use.
        noise = torch.randn_like(x_sub).mul_(0.01)
        noise = F.normalize(noise, dim=1)
        aug = x_sub + noise
        del noise

        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                z_aug = model(aug, sub_edge_index)
                z = model(x_sub, sub_edge_index)

                z_core = z[core_new_index]
                z_aug_core = z_aug[core_new_index]

                loss = 0.5 * (
                        loss_infomax_new(z_core, z_aug_core, core_loss_edges)
                        + loss_infomax_new(z_aug_core, z_core, core_loss_edges)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            z_aug = model(aug, sub_edge_index)
            z = model(x_sub, sub_edge_index)

            z_core = z[core_new_index]
            z_aug_core = z_aug[core_new_index]

            loss = 0.5 * (
                    loss_infomax_new(z_core, z_aug_core, core_loss_edges)
                    + loss_infomax_new(z_aug_core, z_core, core_loss_edges)
            )

            loss.backward()
            optimizer.step()

        loss_value = float(loss.detach().cpu())
        # Release training tensors immediately.
        del aug
        del z_aug
        del z
        del z_core
        del z_aug_core
        del loss

        # ========== Scheduler step：传入训练loss ==========
        scheduler.step(loss_value)

        if train_device.type == 'cuda':
            torch.cuda.empty_cache()

        # ========== 每个epoch都eval（仅用于观察ARI/NMI，不参与早停和scheduler） ==========
        ari, nmi, _ = evaluate_full_graph(
            model=model,
            x_global_cpu=x_global_cpu,
            edge_index_global_cpu=edge_index_global_cpu,
            labels=processed_adata.obs['x'],
            num_cluster=args_cluster_num,
            num_gene=num_gene,
            latent_dim=args_latent_dim,
            eval_device=args.eval_device,
            eval_cluster_method=args.eval_cluster_method,
            seed=args.seed,
            return_embedding=False,
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:3d}: "
            f"loss={loss_value:.4f} | lr={current_lr:.2e} | "
            f"{args.eval_cluster_method} | "
            f"ARI={ari:.4f}, NMI={nmi:.4f} | "
            f"loss_patience: {patience_loss_counter}/{patience_loss}"
        )

        # 记录最优ARI（仅记录日志，不影响早停）
        if ari > best_ari:
            best_ari = ari
            best_nmi = nmi

        # ========== 基于loss的早停逻辑（核心） ==========
        loss_drop = prev_loss - loss_value
        if loss_drop > min_delta_loss:
            # loss下降幅度足够，重置计数器
            patience_loss_counter = 0
        else:
            # loss下降很小或上升，计数器+1
            patience_loss_counter += 1

        if patience_loss_counter >= patience_loss:
            print(
                f"\n==== Loss Early Stop Triggered at epoch {epoch} ===="
                f"\nLoss has been stable for {patience_loss} epochs. Stop training."
            )

            # ========== 仅早停触发时保存一次npz embedding ==========
            if not saved_emb:
                current_ari, current_nmi, emb_np = evaluate_full_graph(
                    model=model,
                    x_global_cpu=x_global_cpu,
                    edge_index_global_cpu=edge_index_global_cpu,
                    labels=processed_adata.obs['x'],
                    num_cluster=args_cluster_num,
                    num_gene=num_gene,
                    latent_dim=args_latent_dim,
                    eval_device=args.eval_device,
                    eval_cluster_method=args.eval_cluster_method,
                    seed=args.seed,
                    return_embedding=True,
                )

                # 自动生成文件名：数据集名 + ARI + NMI + epoch
                save_name = (
                    f"{dataset_name}_earlystop_"
                    f"ari{current_ari:.4f}_nmi{current_nmi:.4f}_"
                    f"epoch{epoch}.npz"
                )
                save_path = os.path.join(output_dir, save_name)
                np.savez(
                    save_path,
                    embedding=emb_np,
                    best_ari=best_ari,
                    best_nmi=best_nmi,
                    stop_epoch=epoch,
                )
                saved_emb = True
                print(f"✅ Embedding saved once at: {save_path}")
            break

        # 更新上一轮loss（只在未触发早停时更新）
        prev_loss = loss_value

        if epoch % 50 == 0:
            clear_cuda()

    elapsed = time.perf_counter() - start_time
    mem_after = process.memory_info().rss

    print('Final Best ARI, NMI:', best_ari, best_nmi)
    print(f"time: {elapsed:.6f}秒")
    print(f"CPU memory before: {mem_before / 1024 / 1024:.2f} MB")
    print(f"CPU memory after:  {mem_after / 1024 / 1024:.2f} MB")

    del model
    del optimizer
    del x_sub
    del sub_edge_index
    del core_loss_edges
    del core_new_index
    del x_global_cpu
    del edge_index_global_cpu

    clear_cuda()


if __name__ == '__main__':
    main()

import pandas as pd
import scipy.sparse as sp
import torch.utils.data
from scipy.sparse.linalg import svds
from setuptools.sandbox import save_path
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import add_self_loops, to_undirected
from scipy.sparse import coo_matrix
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh   
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GATConv, GCNConv
from sklearn.impute import SimpleImputer
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn.cluster import KMeans
import umap
import hdbscan
import scanpy as sc
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import numpy as np
import anndata
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI



#compute correlation for cell
def correlation(data_numpy, k, corr_type='pearson'):

    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    return corr, neighbors

#prepare graph knn or pknn
def prepare_graphs(adata_khvg, dataset_name, save_path,graph_type,k,graph_distance_cutoff_num_stds=0,save_graph=False):

    if graph_type == 'KNN':
        print('Computing KNN graph by scanpy...')
        # use package scanpy to compute knn graph
        distances_csr_matrix = \
            sc.pp.neighbors(adata_khvg, n_neighbors=k + 1, knn=True, copy=True).obsp[
                'distances']
        # ndarray
        distances = distances_csr_matrix.A
        # resize
        neighbors = np.resize(distances_csr_matrix.indices, new_shape=(distances.shape[0], k))

    elif graph_type == 'PKNN':
        print('Computing PKNN graph...')
        if isinstance(adata_khvg.X, np.ndarray):
            X_khvg = adata_khvg.X
        else:
            X_khvg = adata_khvg.X.toarray()
        distances, neighbors = correlation(data_numpy=X_khvg, k=k + 1)

    if graph_distance_cutoff_num_stds:
        cutoff = np.mean(np.nonzero(distances), axis=None) + float(graph_distance_cutoff_num_stds) * np.std(
            np.nonzero(distances), axis=None)
    # shape: 2 * (the number of edge)
    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                if graph_distance_cutoff_num_stds:
                    distance = distances[i][j]
                    if distance < cutoff:
                        if i != neighbors[i][j]:
                            edgelist.append(pair)
                else:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)

    # save
    if save_graph:
        num_hvg = adata_khvg.shape[1]
        k_file = k
        if graph_type == 'KNN':
            graph_name = 'Scanpy'
        elif graph_type == 'PKNN':
            graph_name = 'Pearson'

        filename = f'{dataset_name}.txt'

        final_path = os.path.join(save_path, filename)
        print(f'Saving graph to {final_path}...')
        with open(final_path, 'w') as f:
            edges = [' '.join(e) + '\n' for e in edgelist]
            f.writelines(edges)

    return edgelist

# do load graph
def load_graph(edge_path):
    edgelist = []
    with open(edge_path, 'r') as edge_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edge_file.readlines()]
    return edgelist


def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics["ARI"] = ARI(y_true, y_pred)
    metrics["NMI"] = NMI(y_true, y_pred)
    return metrics

def cluster(embedding, gt_labels, num_cluster,cluster_name= 'kmeans',seed=42):
    if torch.is_tensor(embedding):#新加的这两行
        embedding = embedding.detach().cpu().numpy()

    imputer = SimpleImputer(strategy='mean')
    embedding = imputer.fit_transform(embedding)

    if cluster_name == 'kmeans':
        pd_labels = KMeans(n_clusters=num_cluster, random_state=seed).fit(embedding).labels_
        eval_supervised_metrics = compute_metrics(gt_labels, pd_labels)
        return eval_supervised_metrics, pd_labels

    if cluster_name == 'hdbscan':
        umap_reducer = umap.UMAP()
        u = umap_reducer.fit_transform(embedding)
        cl_sizes = [10, 25, 50, 100]
        min_samples = [5, 10, 25, 50]
        hdbscan_dict = {}
        ari_dict = {}
        for cl_size in cl_sizes:
            for min_sample in min_samples:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=cl_size, min_samples=min_sample)
                clusterer.fit(u)
                ari_dict[(cl_size, min_sample)] = compute_metrics(gt_labels, clusterer.labels_)
                hdbscan_dict[(cl_size, min_sample)] = clusterer.labels_
        max_tuple = max(ari_dict, key=lambda x: ari_dict[x]['ARI'])
        return ari_dict[max_tuple], hdbscan_dict[max_tuple]

def cal_centrality(X, labels):

    centrality_scores = [
        torch.norm(X[labels == label], dim=1) if (labels == label).any() else 0
        for label in torch.unique(labels)
    ]
    return centrality_scores

def spec_clustering(adj_csr, k, batch_size=100):

    degree_matrix = sp.diags(np.array(adj_csr.sum(axis=1)).flatten())  # 度矩阵，也没用上啊
    norm_laplacian = sp.csgraph.laplacian(adj_csr, normed=True)  # 标准化后的拉普拉斯矩阵

   
    u, s, vt = svds(norm_laplacian, k=k)
    eigvals = s[::-1]
    eigvecs = u[:, ::-1]
    # eigvals and eigvecs should be returned

    X = torch.tensor(eigvecs[:, :k], dtype=torch.float32)
    scaler = StandardScaler()  # 归一化
    X_scaled = scaler.fit_transform(X.numpy())

 
    #minibatch_kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42, n_init=10)
    minibatch_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = minibatch_kmeans.fit_predict(X_scaled)
    #print("labels",labels)
    labels = torch.tensor(labels, dtype=torch.int64)
    # labels should be returned

    # 找到每个簇的label，再找到簇内所有节点cluster_indices
    centrality_scores = cal_centrality(X, labels)
    #print('centrality_scores',len(centrality_scores),centrality_scores[0])
    centers = []
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        cluster_indices = torch.where(labels == label)[0]
        if len(cluster_indices) > 0:
            cluster_centrality_scores = centrality_scores[label.item()]
            max_centrality_index = cluster_indices[torch.argmax(cluster_centrality_scores)]
            centers.append(max_centrality_index.item())

    return torch.tensor(centers, dtype=torch.int64)  # ，返回的是个索引列表！！返回所有簇中心的节点索引的张量。

def spec_clustering_new(adj_csr, k, feature_matrix, N, batch_size=100):


    if not isinstance(feature_matrix, torch.Tensor):
        feature_matrix = torch.tensor(
            feature_matrix.toarray() if hasattr(feature_matrix, 'toarray') else feature_matrix,
            dtype=torch.float32)


    norm_laplacian = sp.csgraph.laplacian(adj_csr, normed=True)
    u, s, vt = svds(norm_laplacian, k=k)
    eigvals = s[::-1]
    eigvecs = u[:, ::-1]


    X = torch.tensor(eigvecs[:, :k], dtype=torch.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.numpy())

   
    minibatch_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = minibatch_kmeans.fit_predict(X_scaled)
    labels = torch.tensor(labels, dtype=torch.int64)

   
    centrality_scores = cal_centrality(X, labels)

    selected_indices = []
    selected_features = []
    unique_labels = torch.unique(labels)

    for label in unique_labels:
        cluster_indices = torch.where(labels == label)[0]
        if len(cluster_indices) == 0:
            continue


        cluster_centrality = centrality_scores[label.item()]

      
        sorted_idx = torch.argsort(cluster_centrality, descending=True)
        top_n_indices = sorted_idx[:min(N, len(sorted_idx))]

      
        cluster_node_indices = cluster_indices[top_n_indices]

    
        cluster_features = feature_matrix[cluster_node_indices]


        selected_indices.append(cluster_node_indices)
        selected_features.append(cluster_features)


    if selected_indices:
        all_indices = torch.cat(selected_indices)
        all_features = torch.cat(selected_features, dim=0)
    else:
        all_indices = torch.tensor([], dtype=torch.long)
        all_features = torch.tensor([], dtype=torch.float32)

    return all_indices, all_features






import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='prepare graph!')

    parser.add_argument('--graph_type', type=str, default='PKNN', choices=['KNN', 'PKNN'],
                        help='Type of graph to construct')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors for KNN graph')

    parser.add_argument('--input_h5ad_path', type=str, default="./data/deng_preprocessed.h5ad", help='path to input h5ad file')

    parser.add_argument('--save_graph_path', type=str, default="./graph",help='path to save graph')
    return parser.parse_args()
def main():
    args=parse_arguments()

    h5ad_data_file = args.input_h5ad_path
    args_graph_type = args.graph_type
    args_KNNs_K = args.k


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


    processed_adata = sc.read_h5ad(h5ad_data_file)
    num_nodes, num_gene = processed_adata.shape


    edgelist = prepare_graphs(
    processed_adata,
    dataset_name='muraro',
    save_path=args.save_graph_path,
    graph_type=args_graph_type,
    k=args_KNNs_K,
    save_graph=True,
    )


if __name__ == "__main__":

    main()

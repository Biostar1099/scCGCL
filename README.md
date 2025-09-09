# **scCGCL : Core Cells Sampling-Enhanced Graph Contrastive Learning for Single-cell RNA-seq Clustering**

## Introduction

 In this study, we propose scCGCL-core cells sampling-enhanced graph contrastive learning for single-cell RNA-seq clustering. First, core cells sampling strategy is used to select a set of core cells for each cluster identified by the K-means algorithm. Second, a graph attention network (GAT) is employed to aggregate neighborhood information, thereby capturing intercellular similarity structures. Third, we propose a novel negative sample selection mechanism that promotes inter-cluster separation among cells, guided by preliminary K-means clustering. We evaluated scCGCL across nine scRNA-seq datasets. Extensive experiments show that scCGCL achieves superior clustering performance and high computational efficiency compared with state-of-the-art (SOTA) methods.

## Installation

The package can be installed by `git clone https://github.com/Biostar/scCGCL.git`. The testing setup involves a Windows operating system with 64GB of RAM, powered by an NVIDIA GeForce RTX 4090 GPU 

### Utilize a virtual environment using Anaconda

You can set up the primary environment for scCGCL by using the following command:

```
conda env create -f environment.yml
conda activate sc
```

## Running example

### 1. Collect Dataset.

All data used in this study were obtained from publicly available datasets. Our sample dataset can be accessed through the following link:  
[Google Drive - Dataset](https://drive.google.com/drive/folders/1S4AsPQp-h8wQruj60tfbglCUYvinZFVC?usp=drive_link)



### 2. Generate Preprocessed H5AD File.

```python
python preprocess_data.py --input_h5_path="data/deng.h5" --save_h5ad_dir="data" --filter --norm --log --scale --select_hvg
```

### 3. Prepare Graph

```python
python prepare_graph.py --input_h5ad_path="datadeng_preprocessed.h5ad" ----save_graph_path="./graph"
```

### 4. Apply scCGCL

```python
python main.py --input_h5ad_path="data/preprocessed/deng_preprocessed.h5ad" --cluster_num 6 --save_graph_path="./graph/knn_graph.txt"
```

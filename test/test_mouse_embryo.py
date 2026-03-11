import os
import torch
import pandas as pd
import scanpy as sc
from matplotlib.pyplot import title
import argparse
from mymodel.preprocess import construct_neighbor_graph,lsi
from sipbuild.generator.outputs import output_api
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score
from mymodel.utils import read_list_from_file
import matplotlib.pyplot as plt
from mymodel.dataloder import SpatialOmicsDataLoader
import numpy as np
import random
import mymodel.utils as u
print("USING UTILS:", u.__file__)
print("mclust_R SOURCE LINE:", u.mclust_R.__code__.co_firstlineno)

# 设置随机种子确保可重复性
def setup_seed(seed=2020):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设置全局随机种子
setup_seed(2020)

# read data
file_fold = './data/' #please replace 'file_fold' with the download path

adata_omics1 = sc.read_h5ad(file_fold + 's1_adata_rna.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 's1_adata_atac.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

meta = pd.read_csv('./data/meta.tsv', sep='\t')
adata_omics1.obs.index = adata_omics1.obs.index.str.replace("mult-","",regex=False)
extra_barcodes = set(meta['barcode']) - set(adata_omics1.obs.index)
meta = meta[~meta['barcode'].isin(extra_barcodes)]
adata_omics1.obs = meta
adata_omics2.obs = meta
ground_truth = meta['Joint_clusters']


parser = argparse.ArgumentParser(description='空间多组学数据训练')
parser.add_argument('--batch_size', default=2186, type=int)
parser.add_argument("--temperature_f", default=0.9)
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--weight_decay", default=0.0001)
parser.add_argument("--rec_epochs", default=200)
parser.add_argument("--fine_tune_epochs", default=120)
parser.add_argument("--low_feature_dim", default=256)
parser.add_argument("--high_feature_dim", default=768)
parser.add_argument("--cluster_loss_weight", default=0.3)
parser.add_argument("--alpha", default=0.5)
parser.add_argument("--walk_steps", default=7)
parser.add_argument("--use_adaptive_fusion", default=True)
parser.add_argument("--fusion_hidden_dim", default=256)
parser.add_argument("--fusion_dropout", default=0.0)
parser.add_argument('--n_clusters', default=14, type=int, help='聚类数量')
parser.add_argument('--feature_weight', type=float, default=None, help='特征图的融合权重，不设置则使用自适应权重')
parser.add_argument('--spatial_weight', type=float, default=None, help='空间图的融合权重，不设置则使用自适应权重')
parser.add_argument('--trrust_file', 
                   type=str, 
                   default='E:/code/mymodel2/data/Mouse_Brain/trrust_rawdata.mouse.tsv', 
                   help='TRRUST数据库文件路径')
parser.add_argument('--peak_tf_db', 
                   type=str, 
                   default='E:/code/mymodel2/data/mouse_embryo/peaks_to_tf_me.bed', 
                   help='峰-TF数据库文件路径，支持BED/CSV/JSON格式')
parser.add_argument('--datatype',
                   type=str,
                   default='SPOTS',
                   choices=['SPOTS', 'Spatial-epigenome-transcriptome'],
                   help='数据类型，可选SPOTS或Spatial-epigenome-transcriptome')
args = parser.parse_args()


# 初始化数据加载器
data_loader = SpatialOmicsDataLoader(
    data_dir='E:/code/mymodel2/data/mouse_embryo/',
    trrust_file_path=args.trrust_file,
    peak_tf_db_path=args.peak_tf_db,
    enable_context_filtering=False
)


# 使用GRN填补数据
adata_omics1= data_loader.impute_rna_with_grn(
    adata_omics1, 
    adata_omics2,
    confidence_threshold=0.8,
    min_cells=4,
)
# 在数据预处理之前应用填补
print("应用ATAC数据填补...")
adata_omics2= data_loader.impute_atac_with_peak_tf(
    adata_omics1, 
    adata_omics2,
    confidence_threshold=0.1,
    min_cells=30
)



from mymodel.preprocess import pca ,lsi
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)
adata_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_high, n_comps=min(50, adata_omics2.n_vars-1))    
    
if 'X_lsi' not in adata_omics2.obsm.keys():
    sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
    lsi(adata_omics2, use_highly_variable=False, n_components=51)
adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
    

data = construct_neighbor_graph(adata_omics1, adata_omics2)
from mymodel.utils import clustering

setup_seed(2020)
from mymodel.test_train import SpatialOmicsTrainer
trainer = SpatialOmicsTrainer(args, data)
output = trainer.train()


adata = adata_omics1.copy()  
adata.obsm['DeepSpak'] = output['emb_combined'].copy()
# 聚类
from mymodel.utils import clustering
tool = 'mclust'  # mclust, leiden, and louvain

# 对每个嵌入进行聚类
clustering(adata, key='DeepSpak', add_key='DeepSpak', n_clusters=args.n_clusters, method=tool, use_pca=True)
Our_ari = adjusted_rand_score(adata.obs['DeepSpak'], ground_truth)
print(f"Combined embedding (ARI): {Our_ari:.6f}")




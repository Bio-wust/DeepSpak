import os
import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import networkx as nx
from tqdm import tqdm
import json
import torch
import anndata as ad
from mymodel.preprocess import construct_neighbor_graph, lsi, pca
# 导入蛋白质ID映射器
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mymodel.protein_id_mapper import ProteinIDMapper
import random
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


class SpatialOmicsDataLoader:
    """
    空间多组学数据加载器，包含GRN（基因调控网络）预处理功能
    
    主要功能:
    1. 加载RNA和ATAC数据
    2. 使用GRN预测RNA和ATAC数据
    3. 将原始数据和预测数据分别存储在AnnData对象中
    """
    
    def __init__(self, 
                 data_dir,
                 trrust_file_path=None,
                 peak_tf_db_path=None,
                 datatype='Spatial-epigenome-transcriptome',
                 force_reprocess=False,
                 cache_dir=None,
                 string_protein_info_file=None,
                 ppi_file_path=None,
                 enable_protein_features=None,
                 enable_context_filtering=True):
        """
        初始化数据加载器
        
        参数:
            data_dir: 数据目录
            trrust_file_path: TRRUST数据库文件路径
            peak_tf_db_path: 峰-TF数据库文件路径
            datatype: 数据类型，可选'SPOTS'或'Spatial-epigenome-transcriptome'
            force_reprocess: 是否强制重新处理数据
            cache_dir: 缓存目录，默认为data_dir下的cache子目录
            string_protein_info_file: STRING数据库蛋白质信息文件路径，用于蛋白质ID映射
            ppi_file_path: PPI网络文件路径，用于蛋白质表达数据填补
            enable_protein_features: 是否启用蛋白质相关功能，None为自动检测
            enable_context_filtering: 是否启用上下文基因过滤，默认为True
        """
        self.data_dir = data_dir
        self.trrust_file_path = trrust_file_path
        self.peak_tf_db_path = peak_tf_db_path
        self.datatype = datatype
        self.force_reprocess = force_reprocess
        self.cache_dir = cache_dir or os.path.join(data_dir, 'cache')
        self.enable_context_filtering = enable_context_filtering
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # TF-基因调控网络
        self.tf_gene_network = None
        self.peak_tf_map = None
        
        # 智能判断是否需要蛋白质功能
        if enable_protein_features is None:
            # 自动检测：如果提供了蛋白质相关文件路径，或者datatype暗示有蛋白质数据，则启用
            self.enable_protein_features = (
                (string_protein_info_file is not None) or 
                (ppi_file_path is not None) or 
                (datatype in ['SPOTS', 'Spatial-proteome-transcriptome']) or
                os.path.exists(os.path.join(data_dir, '9606.protein.info.v12.0.txt')) or
                os.path.exists(os.path.join(data_dir, '9606.protein.links.v12.0.txt'))
            )
        else:
            self.enable_protein_features = enable_protein_features
        
        print(f"蛋白质功能启用状态: {self.enable_protein_features}")
        
        # 添加蛋白质ID映射器和PPI数据库路径（仅在启用时）
        if self.enable_protein_features:
            self.string_protein_info_file = string_protein_info_file or os.path.join(data_dir, '9606.protein.info.v12.0.txt')
            self.ppi_file_path = ppi_file_path or os.path.join(data_dir, '9606.protein.links.v12.0.txt')  # PPI网络文件
            
            # 打印蛋白质功能相关文件状态
            print(f"PPI网络文件路径: {self.ppi_file_path}")
            print(f"STRING蛋白质信息文件路径: {self.string_protein_info_file}")
        else:
            self.string_protein_info_file = None
            self.ppi_file_path = None
            print("蛋白质功能未启用，PPI网络和STRING数据库功能将不可用")
        
        # 初始化缓存文件路径
        self.ppi_network_cache = os.path.join(self.cache_dir, 'ppi_network.pkl')
        self.protein_mapping_cache = os.path.join(self.cache_dir, 'protein_mapping.pkl')
        
        # 初始化PPI网络和蛋白质映射缓存
        self.ppi_network = None
        self.protein_id_mapper = None
        self._cached_string_ids = {}  # 用于缓存STRING ID映射结果
        self._cached_adt_mapping = {}  # 用于缓存ADT映射结果
        self._cached_processed_adt = {}  # 用于缓存处理后的ADT数据，键为置信度阈值
        
        # 从缓存加载或初始化蛋白质ID映射器（仅在启用时）
        if self.enable_protein_features:
            if not self.force_reprocess and os.path.exists(self.protein_mapping_cache):
                try:
                    print(f"从缓存加载蛋白质ID映射器: {self.protein_mapping_cache}")
                    import pickle
                    with open(self.protein_mapping_cache, 'rb') as f:
                        self.protein_id_mapper = pickle.load(f)
                    print("成功加载蛋白质ID映射器")
                except Exception as e:
                    print(f"加载蛋白质ID映射器缓存失败: {str(e)}")
                    self._initialize_protein_id_mapper()
            else:
                self._initialize_protein_id_mapper()
        else:
            print("跳过蛋白质ID映射器初始化（未启用蛋白质功能）")
            self.protein_id_mapper = None
        
        # TF-基因调控网络延迟加载（需要adata_rna参数）
        self.tf_gene_network = None
        self._tf_gene_network_loaded = False
            
        # 如果提供了峰-TF数据库文件，加载峰-TF映射
        if self.peak_tf_db_path and os.path.exists(self.peak_tf_db_path):
            self.peak_tf_map = self._load_peak_tf_map()
    
    def _initialize_protein_id_mapper(self):
        """初始化蛋白质ID映射器并保存到缓存，只处理与ADT数据集相关的蛋白质信息"""
        # 首先创建ProteinIDMapper对象
        print("创建蛋白质ID映射器...")
        self.protein_id_mapper = ProteinIDMapper(string_protein_info_file=self.string_protein_info_file)
        
        # 获取ADT数据相关的蛋白质名称 - 应该从实际ADT数据中获取，而不只是从手动映射
        relevant_proteins = set()
        
        # 首先从手动映射中获取
        for protein, _ in self.protein_id_mapper.manual_mapping.items():
            relevant_proteins.add(protein)
            
        # TODO: 应该从实际的ADT数据中获取所有蛋白质名称
        # 目前暂时使用已知的ADT蛋白质列表
        adt_proteins = ['CD163', 'CR2', 'PCNA', 'VIM', 'KRT5', 'CD68', 'CEACAM8', 'PTPRC', 'HLA-DRA', 'PAX5',
                       'SDC1', 'CD8A', 'BCL2', 'CD19', 'PDCD1', 'ACTA2', 'FCGR3A', 'ITGAX', 'CXCR5', 'EPCAM', 
                       'MS4A1', 'CD3E', 'CD14', 'CD40', 'PECAM1', 'CD4', 'ITGAM', 'CD27', 'CCR7', 'CD274']
        relevant_proteins.update(adt_proteins)
        
        print(f"找到 {len(relevant_proteins)} 个与ADT数据相关的蛋白质名称")
        
        # STRING数据库信息已经在protein_id_mapper初始化时加载，这里只需要提取基因符号
        if hasattr(self.protein_id_mapper, 'string_protein_info') and self.protein_id_mapper.string_protein_info:
            print(f"STRING数据库已在映射器中加载: {len(self.protein_id_mapper.string_protein_info)}个蛋白质")
            
            # 提取基因符号，增强映射能力
            if not hasattr(self.protein_id_mapper, 'gene_symbol_to_string_ids'):
                self.protein_id_mapper.extract_gene_symbols_from_annotation()
        else:
            print("警告: STRING数据库未在映射器中加载，映射功能可能受限")
        
        # 保存蛋白质ID映射器到缓存
        self._save_protein_mapper_cache()
    
    def _save_protein_mapper_cache(self):
        """保存蛋白质ID映射器到缓存"""
        try:
            print(f"保存蛋白质ID映射器到缓存: {self.protein_mapping_cache}")
            import pickle
            with open(self.protein_mapping_cache, 'wb') as f:
                pickle.dump(self.protein_id_mapper, f)
            print("蛋白质ID映射器已保存到缓存")
        except Exception as e:
            print(f"保存蛋白质ID映射器缓存失败: {str(e)}")
    
    def _get_context_genes(self, adata, min_cells=5, min_expr=1.0):
        """获取在数据中确实表达的上下文基因集"""
        X = adata.raw.X if adata.raw is not None else adata.X
        
        if scipy.sparse.issparse(X):  # 稀疏矩阵
            gene_counts = (X > min_expr).sum(axis=0)
        else:  # 密集矩阵
            gene_counts = (X > min_expr).sum(axis=0)
        
        # 强制转换为1D numpy数组（解决matrix vs array问题）
        gene_counts = np.asarray(gene_counts).flatten()
        mask = gene_counts >= min_cells
        
        return set(adata.var_names[mask])
    
    def _ensure_tf_gene_network(self, adata_rna):
        """确保TF-基因网络已加载（延迟加载）"""
        if not self._tf_gene_network_loaded:
            if self.trrust_file_path and os.path.exists(self.trrust_file_path):
                print("首次使用，加载TF-基因调控网络...")
                self.tf_gene_network = self._load_tf_gene_network(adata_rna)
                self._tf_gene_network_loaded = True
            else:
                print("警告: 未找到TRRUST文件，无法加载TF-基因调控网络")
                self.tf_gene_network = None
                self._tf_gene_network_loaded = True
    
    def _load_tf_gene_network(self, adata_rna=None):
        """加载TF-基因调控网络，使用上下文基因过滤"""
        print(f"加载TF-基因调控网络: {self.trrust_file_path}")
        tf_gene_dict = {}
        
        try:
            # 读取TRRUST数据
            # 格式: TF  target  interaction_type  PMID
            trrust_df = pd.read_csv(self.trrust_file_path, sep='\t', header=None)
            
            if trrust_df.shape[1] >= 2:
                # 标准化列名，兼容不同格式
                if trrust_df.shape[1] >= 4:
                    trrust_df.columns = ['TF', 'Target', 'Mode', 'PMID']
                elif trrust_df.shape[1] == 3:
                    trrust_df.columns = ['TF', 'Target', 'Mode']
                else:
                    trrust_df.columns = ['TF', 'Target']
                
                print(f"原始TRRUST网络: {len(trrust_df):,}条边, {trrust_df['TF'].nunique()}个TF")
                
                # 根据设置决定是否使用上下文基因过滤
                if self.enable_context_filtering and adata_rna is not None:
                    print("应用上下文基因过滤...")
                    
                    # 根据数据类型自适应调整参数
                    n_cells = adata_rna.n_obs
                    if n_cells < 1000:  # 小数据集
                        min_cells, min_expr = 5, 0.5
                    elif n_cells < 5000:  # 中等数据集
                        min_cells, min_expr = 10, 0.5
                    else:  # 大数据集
                        min_cells, min_expr = 20, 1.0
                    
                    print(f"  数据集大小: {n_cells}细胞, 使用参数: min_cells={min_cells}, min_expr={min_expr}")
                    
                    # 获取上下文基因
                    ctx_genes = self._get_context_genes(adata_rna, min_cells=min_cells, min_expr=min_expr)
                    print(f"  上下文基因数: {len(ctx_genes):,}")
                    
                    # 过滤网络：只保留TF和Target都在上下文基因中的边
                    trrust_ctx = trrust_df[
                        trrust_df["TF"].isin(ctx_genes) & 
                        trrust_df["Target"].isin(ctx_genes)
                    ].reset_index(drop=True)
                    
                    print(f"  上下文过滤后: {len(trrust_ctx):,}条边 (保留 {len(trrust_ctx)/len(trrust_df):.1%})")
                    
                    # 可选：移除超级hub TF
                    tf_degrees = trrust_ctx["TF"].value_counts()
                    max_degree = min(500, max(50, len(ctx_genes) // 20))  # 动态调整hub阈值
                    hub_tfs = tf_degrees[tf_degrees > max_degree].index
                    
                    if len(hub_tfs) > 0:
                        trrust_ctx = trrust_ctx[~trrust_ctx["TF"].isin(hub_tfs)]
                        print(f"  移除{len(hub_tfs)}个超级hub TF (degree > {max_degree})")
                        print(f"  最终网络: {len(trrust_ctx):,}条边")
                    
                    # 使用过滤后的网络
                    filtered_df = trrust_ctx
                elif adata_rna is not None:
                    print("跳过上下文基因过滤 (enable_context_filtering=False)")
                    filtered_df = trrust_df
                else:
                    print("警告: 未提供adata_rna，跳过上下文基因过滤")
                    filtered_df = trrust_df
                
                # 构建字典格式的网络
                for _, row in filtered_df.iterrows():
                    tf = row['TF']
                    target = row['Target']
                    
                    if tf not in tf_gene_dict:
                        tf_gene_dict[tf] = []
                    
                    if target not in tf_gene_dict[tf]:
                        tf_gene_dict[tf].append(target)
                
                print(f"最终TF-基因网络: {len(tf_gene_dict)}个TF, {sum(len(genes) for genes in tf_gene_dict.values())}个调控关系")
                print(f"上下文基因过滤状态: {'已启用' if self.enable_context_filtering and adata_rna is not None else '已禁用'}")
                return tf_gene_dict
            else:
                print(f"TRRUST文件格式不正确，列数: {trrust_df.shape[1]}")
                return None
        except Exception as e:
            print(f"加载TF-基因调控网络出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_peak_tf_map(self):
        """加载峰-TF映射关系"""
        print(f"加载峰-TF映射关系: {self.peak_tf_db_path}")
        peak_tf_map = {}
        
        try:
            # 根据文件扩展名确定加载方式
            if self.peak_tf_db_path.endswith('.bed'):
                # BED格式: chrom start end name score strand TFs
                df = pd.read_csv(self.peak_tf_db_path, sep='\t', header=None)
                
                if df.shape[1] >= 7:  # 至少需要7列
                    for _, row in df.iterrows():
                        peak_id = f"{row[0]}-{row[1]}-{row[2]}"  # chrom-start-end
                        tfs = row[6].split(',') if isinstance(row[6], str) else []
                        if tfs:
                            peak_tf_map[peak_id] = tfs
                else:
                    print(f"BED文件格式不正确，列数: {df.shape[1]}")
            
            elif self.peak_tf_db_path.endswith('.csv'):
                # CSV格式: peak_id,tf1,tf2,...
                df = pd.read_csv(self.peak_tf_db_path)
                
                if 'peak_id' in df.columns and 'tfs' in df.columns:
                    for _, row in df.iterrows():
                        peak_id = row['peak_id']
                        tfs = row['tfs'].split(',') if isinstance(row['tfs'], str) else []
                        if tfs:
                            peak_tf_map[peak_id] = tfs
                else:
                    print(f"CSV文件缺少必要列: peak_id或tfs")
            
            elif self.peak_tf_db_path.endswith('.json'):
                # JSON格式: {"peak_id": ["tf1", "tf2", ...]}
                with open(self.peak_tf_db_path, 'r') as f:
                    peak_tf_map = json.load(f)
            
            else:
                print(f"不支持的文件格式: {self.peak_tf_db_path}")
                return None
            
            print(f"成功加载峰-TF映射关系: {len(peak_tf_map)} 个峰, {sum(len(tfs) for tfs in peak_tf_map.values())} 个关联")
            return peak_tf_map
        
        except Exception as e:
            print(f"加载峰-TF映射关系出错: {str(e)}")
            return None
    
    def predict_rna_from_grn(self, adata_omics1, adata_omics2):
        """
        使用基因调控网络预测RNA表达数据
        
        参数:
            adata_omics1: RNA数据的AnnData对象
            adata_omics2: ATAC数据的AnnData对象
            
        返回:
            更新后的adata_omics1，包含预测的表达数据
        """
        print("使用GRN预测RNA表达数据...")
        
        # 确保TF-基因网络已加载（使用上下文基因过滤）
        self._ensure_tf_gene_network(adata_omics1)
        
        if self.tf_gene_network is None:
            print("警告: 未加载TF-基因调控网络，无法预测RNA表达")
            # 保存原始数据用于后续处理
            adata_omics1.obsm['original_expression'] = adata_omics1.X.copy()
            adata_omics1.obsm['predicted_expression'] = adata_omics1.X.copy()
            return adata_omics1
        
        # 构建GRN网络
        G = nx.DiGraph()
        
        # 添加TF-目标基因边
        for tf, targets in self.tf_gene_network.items():
            for target in targets:
                G.add_edge(tf, target)
        
        # 获取所有基因
        all_genes = set(adata_omics1.var_names)
        
        # 获取网络中的基因
        network_genes = set(G.nodes())
        
        # 找出数据集和网络中共有的基因
        common_genes = all_genes.intersection(network_genes)
        print(f"数据集中的基因: {len(all_genes)}, 网络中的基因: {len(network_genes)}, 共有基因: {len(common_genes)}")
        
        # 创建基因到索引的映射
        gene_to_idx = {gene: i for i, gene in enumerate(adata_omics1.var_names)}
        
        # 获取原始表达矩阵
        X_orig = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X
        
        # 初始化预测表达矩阵，从原始数据复制以保证非零值
        X_pred = X_orig.copy()
        
        # 对每个TF，预测其目标基因的表达
        for tf, targets in tqdm(self.tf_gene_network.items(), desc="预测基因表达"):
            if tf in gene_to_idx:
                tf_idx = gene_to_idx[tf]
                tf_expr = X_orig[:, tf_idx]
                
                for target in targets:
                    if target in gene_to_idx:
                        target_idx = gene_to_idx[target]
                        # 扩展模型：目标基因表达受TF表达影响，但保留原始表达的一部分特征
                        # 使用一个调整因子来混合原始表达和基于TF的表达
                        tf_weight = 0.7  # TF表达的影响权重
                        orig_weight = 0.3  # 原始表达的保留权重
                        
                        # 原始表达值
                        orig_expr = X_orig[:, target_idx]
                        
                        # 更新表达，确保有合理的非零值
                        X_pred[:, target_idx] = orig_weight * orig_expr + tf_weight * tf_expr
        
        # 存储原始和预测的表达矩阵
        adata_omics1.obsm['original_expression'] = scipy.sparse.csr_matrix(X_orig)
        adata_omics1.obsm['predicted_expression'] = scipy.sparse.csr_matrix(X_pred)
        
        print("RNA表达预测完成")
        print(f"预测矩阵非零元素数: {np.count_nonzero(X_pred)}")
        print(f"预测矩阵均值: {np.mean(X_pred)}")
        
        return adata_omics1
    
    def predict_atac_from_peak_tf(self, adata_omics1, adata_omics2):
        """
        使用峰-TF映射关系预测ATAC数据
        
        参数:
            adata_omics1: RNA数据的AnnData对象
            adata_omics2: ATAC数据的AnnData对象
            
        返回:
            更新后的adata_omics2，包含预测的ATAC数据
        """
        print("使用峰-TF映射预测ATAC数据...")
        
        if self.peak_tf_map is None or self.tf_gene_network is None:
            print("警告: 未加载峰-TF映射或TF-基因网络，无法预测ATAC数据")
            # 保存原始数据用于后续处理
            adata_omics2.obsm['original_atac'] = adata_omics2.X.copy()
            adata_omics2.obsm['predicted_atac'] = adata_omics2.X.copy()
            return adata_omics2
        
        # 获取峰的ID
        peak_ids = []
        for peak_name in adata_omics2.var_names:
            # 尝试从峰名称中提取染色体-开始-结束格式
            parts = peak_name.split(':')
            if len(parts) >= 2 and '-' in parts[1]:
                chrom = parts[0]
                pos = parts[1].split('-')
                if len(pos) >= 2:
                    start, end = pos[0], pos[1]
                    peak_id = f"{chrom}-{start}-{end}"
                    peak_ids.append(peak_id)
            else:
                # 使用原始名称作为ID
                peak_ids.append(peak_name)
        
        # 创建峰到索引的映射
        peak_to_idx = {peak: i for i, peak in enumerate(peak_ids)}
        
        # 创建基因到索引的映射
        gene_to_idx = {gene: i for i, gene in enumerate(adata_omics1.var_names)}
        
        # 获取原始ATAC矩阵
        X_atac = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else adata_omics2.X
        
        # 获取原始RNA矩阵
        X_rna = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X
        
        # 初始化预测ATAC矩阵，从原始数据复制以保证非零值
        X_pred_atac = X_atac.copy()
        
        # 对每个峰，根据其调控的TF预测开放状态
        atac_weight = 0.7  # 原始ATAC数据的保留权重
        tf_weight = 0.3  # TF表达的影响权重
        
        processed_peaks = 0
        for peak_id, tfs in tqdm(self.peak_tf_map.items(), desc="预测ATAC数据"):
            if peak_id in peak_to_idx:
                peak_idx = peak_to_idx[peak_id]
                processed_peaks += 1
                
                tf_influence = np.zeros(X_atac.shape[0])
                tf_count = 0
                
                for tf in tfs:
                    if tf in gene_to_idx:
                        tf_idx = gene_to_idx[tf]
                        # 累加所有相关TF的表达
                        tf_influence += X_rna[:, tf_idx]
                        tf_count += 1
                
                if tf_count > 0:
                    # 平均TF表达
                    tf_influence /= tf_count
                    
                    # 更新预测值，混合原始ATAC信号和TF表达的影响
                    X_pred_atac[:, peak_idx] = atac_weight * X_atac[:, peak_idx] + tf_weight * tf_influence
        
        # 如果处理的峰太少，则使用替代策略
        if processed_peaks < 100:
            print(f"警告: 只找到 {processed_peaks} 个匹配的峰，使用替代预测策略")
            
            # 随机抽样一些TF和峰进行关联
            np.random.seed(42)  # 固定随机数种子以确保可重复性
            
            # 假设20%的峰与随机TF关联
            num_peaks = X_atac.shape[1]
            num_to_process = int(num_peaks * 0.2)
            
            # 随机选择一些峰
            peak_indices = np.random.choice(num_peaks, num_to_process, replace=False)
            
            # 获取所有TF
            all_tfs = list(self.tf_gene_network.keys())
            
            for peak_idx in tqdm(peak_indices, desc="替代ATAC预测"):
                # 为每个峰随机选择1-3个TF
                num_tfs = np.random.randint(1, 4)
                selected_tfs = np.random.choice(all_tfs, num_tfs, replace=False)
                
                tf_influence = np.zeros(X_atac.shape[0])
                
                for tf in selected_tfs:
                    if tf in gene_to_idx:
                        tf_idx = gene_to_idx[tf]
                        tf_influence += X_rna[:, tf_idx]
                
                # 平均TF表达
                tf_influence /= num_tfs
                
                # 更新预测值
                X_pred_atac[:, peak_idx] = atac_weight * X_atac[:, peak_idx] + tf_weight * tf_influence
        
        # 存储原始和预测的ATAC矩阵
        adata_omics2.obsm['original_atac'] = scipy.sparse.csr_matrix(X_atac)
        adata_omics2.obsm['predicted_atac'] = scipy.sparse.csr_matrix(X_pred_atac)
        
        print("ATAC数据预测完成")
        print(f"预测矩阵非零元素数: {np.count_nonzero(X_pred_atac)}")
        print(f"预测矩阵均值: {np.mean(X_pred_atac)}")
        
        return adata_omics2
    
    def create_prediction_anndatas(self, adata_omics1, adata_omics2):
        """
        从预测数据创建独立的AnnData对象
        
        参数:
            adata_omics1: 包含RNA预测数据的AnnData对象
            adata_omics2: 包含ATAC预测数据的AnnData对象
            
        返回:
            adata_omics3: RNA预测数据的AnnData对象
            adata_omics4: ATAC预测数据的AnnData对象
        """
        print("从预测数据创建独立的AnnData对象...")
        
        # 检查预测数据是否存在
        if 'predicted_expression' not in adata_omics1.obsm:
            raise ValueError("预测的RNA表达数据不存在，请先运行GRN预测")
        if 'predicted_atac' not in adata_omics2.obsm:
            raise ValueError("预测的ATAC数据不存在，请先运行TF-峰映射预测")
        
        # 获取预测数据
        predicted_rna = adata_omics1.obsm['predicted_expression']
        if scipy.sparse.issparse(predicted_rna):
            # 稀疏矩阵需要特殊处理
            rna_nonzero = predicted_rna.nnz
            rna_mean = predicted_rna.mean()
            print(f"RNA预测数据统计: 非零元素数={rna_nonzero}, 均值={rna_mean}")
        else:
            print(f"RNA预测数据统计: 非零元素数={np.count_nonzero(predicted_rna)}, 均值={np.mean(predicted_rna)}")
            
        predicted_atac = adata_omics2.obsm['predicted_atac']
        if scipy.sparse.issparse(predicted_atac):
            # 稀疏矩阵需要特殊处理
            atac_nonzero = predicted_atac.nnz
            atac_mean = predicted_atac.mean()
            print(f"ATAC预测数据统计: 非零元素数={atac_nonzero}, 均值={atac_mean}")
        else:
            print(f"ATAC预测数据统计: 非零元素数={np.count_nonzero(predicted_atac)}, 均值={np.mean(predicted_atac)}")
        
        # 确保数据不全为零
        if scipy.sparse.issparse(predicted_rna):
            # 对稀疏矩阵使用nnz检查是否全为零
            if predicted_rna.nnz == 0:
                print("警告: RNA预测数据全为零，将使用原始数据")
                original_rna = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X
                predicted_rna = scipy.sparse.csr_matrix(original_rna.copy())
        else:
            if np.count_nonzero(predicted_rna) == 0:
                print("警告: RNA预测数据全为零，将使用原始数据")
                original_rna = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X
                predicted_rna = original_rna.copy()
            
        if scipy.sparse.issparse(predicted_atac):
            # 对稀疏矩阵使用nnz检查是否全为零
            if predicted_atac.nnz == 0:
                print("警告: ATAC预测数据全为零，将使用原始数据")
                original_atac = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else adata_omics2.X
                predicted_atac = scipy.sparse.csr_matrix(original_atac.copy())
        else:
            if np.count_nonzero(predicted_atac) == 0:
                print("警告: ATAC预测数据全为零，将使用原始数据")
                original_atac = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else adata_omics2.X
                predicted_atac = original_atac.copy()
        
        # 检查数据形状
        if scipy.sparse.issparse(predicted_rna):
            if predicted_rna.shape != (adata_omics1.n_obs, adata_omics1.n_vars):
                print(f"警告: RNA预测数据形状不匹配，期望{(adata_omics1.n_obs, adata_omics1.n_vars)}，实际{predicted_rna.shape}")
                # 重新构建预测数据
                original_rna = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X
                predicted_rna = scipy.sparse.csr_matrix(original_rna.copy())
        else:
            if predicted_rna.shape != (adata_omics1.n_obs, adata_omics1.n_vars):
                print(f"警告: RNA预测数据形状不匹配，期望{(adata_omics1.n_obs, adata_omics1.n_vars)}，实际{predicted_rna.shape}")
                # 重新构建预测数据
                original_rna = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X
                predicted_rna = original_rna.copy()
            
        if scipy.sparse.issparse(predicted_atac):
            if predicted_atac.shape != (adata_omics2.n_obs, adata_omics2.n_vars):
                print(f"警告: ATAC预测数据形状不匹配，期望{(adata_omics2.n_obs, adata_omics2.n_vars)}，实际{predicted_atac.shape}")
                # 重新构建预测数据
                original_atac = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else adata_omics2.X
                predicted_atac = scipy.sparse.csr_matrix(original_atac.copy())
        else:
            if predicted_atac.shape != (adata_omics2.n_obs, adata_omics2.n_vars):
                print(f"警告: ATAC预测数据形状不匹配，期望{(adata_omics2.n_obs, adata_omics2.n_vars)}，实际{predicted_atac.shape}")
                # 重新构建预测数据
                original_atac = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else adata_omics2.X
                predicted_atac = original_atac.copy()
            
        # 检查是否有NaN或无穷值
        if not scipy.sparse.issparse(predicted_rna):
            if np.isnan(predicted_rna).any() or np.isinf(predicted_rna).any():
                print("警告: RNA预测数据包含NaN或无穷值，将进行修复")
                predicted_rna = np.nan_to_num(predicted_rna, nan=0.0, posinf=1.0, neginf=0.0)
            
        if not scipy.sparse.issparse(predicted_atac):
            if np.isnan(predicted_atac).any() or np.isinf(predicted_atac).any():
                print("警告: ATAC预测数据包含NaN或无穷值，将进行修复")
                predicted_atac = np.nan_to_num(predicted_atac, nan=0.0, posinf=1.0, neginf=0.0)
        
        try:
            # 创建RNA预测数据的AnnData对象
            # 使用基本参数创建AnnData对象（X, obs, var）
            adata_omics3 = ad.AnnData(X=predicted_rna)
            
            # 复制必要的元数据
            adata_omics3.obs = adata_omics1.obs.copy()
            adata_omics3.var = adata_omics1.var.copy()
            
            # 复制其他重要的数据结构
            if hasattr(adata_omics1, 'uns') and isinstance(adata_omics1.uns, dict):
                adata_omics3.uns = adata_omics1.uns.copy()
                
            # 复制空间信息和其他必要的obsm数据
            if 'spatial' in adata_omics1.obsm:
                adata_omics3.obsm['spatial'] = adata_omics1.obsm['spatial'].copy()
            
            # 创建ATAC预测数据的AnnData对象
            # 同样使用基本参数创建
            adata_omics4 = ad.AnnData(X=predicted_atac)
            
            # 复制必要的元数据
            adata_omics4.obs = adata_omics2.obs.copy()
            adata_omics4.var = adata_omics2.var.copy()
            
            # 复制其他重要的数据结构
            if hasattr(adata_omics2, 'uns') and isinstance(adata_omics2.uns, dict):
                adata_omics4.uns = adata_omics2.uns.copy()
                
            # 复制空间信息
            if 'spatial' in adata_omics2.obsm:
                adata_omics4.obsm['spatial'] = adata_omics2.obsm['spatial'].copy()
            
            # 确保观测样本一致
            if not set(adata_omics3.obs_names).issubset(set(adata_omics4.obs_names)):
                print("警告: RNA预测数据和ATAC预测数据的观测样本不一致")
                # 取交集
                common_obs = sorted(list(set(adata_omics3.obs_names).intersection(set(adata_omics4.obs_names))))
                if len(common_obs) == 0:
                    raise ValueError("RNA预测数据和ATAC预测数据没有共同的观测样本")
                print(f"使用{len(common_obs)}个共同的观测样本")
                adata_omics3 = adata_omics3[common_obs].copy()
                adata_omics4 = adata_omics4[common_obs].copy()
            else:
                # 确保顺序一致
                adata_omics4 = adata_omics4[adata_omics3.obs_names].copy()
            
            # 验证对象内容
            if scipy.sparse.issparse(adata_omics3.X):
                rna_nonzero = adata_omics3.X.nnz
                rna_mean = adata_omics3.X.mean()
                print(f"RNA预测数据AnnData对象X统计: 非零元素数={rna_nonzero}, 均值={rna_mean}")
            else:
                print(f"RNA预测数据AnnData对象X统计: 非零元素数={np.count_nonzero(adata_omics3.X)}, 均值={np.mean(adata_omics3.X)}")
                
            if scipy.sparse.issparse(adata_omics4.X):
                atac_nonzero = adata_omics4.X.nnz
                atac_mean = adata_omics4.X.mean()
                print(f"ATAC预测数据AnnData对象X统计: 非零元素数={atac_nonzero}, 均值={atac_mean}")
            else:
                print(f"ATAC预测数据AnnData对象X统计: 非零元素数={np.count_nonzero(adata_omics4.X)}, 均值={np.mean(adata_omics4.X)}")
            
            # 打印形状信息
            print(f"成功创建独立的预测数据AnnData对象:")
            print(f"- RNA预测数据 (adata_omics3): {adata_omics3.shape}")
            print(f"- ATAC预测数据 (adata_omics4): {adata_omics4.shape}")
            
            # 最终再次检查数据有效性
            if adata_omics3.n_obs == 0 or adata_omics3.n_vars == 0:
                raise ValueError("RNA预测数据AnnData对象维度为零")
            if adata_omics4.n_obs == 0 or adata_omics4.n_vars == 0:
                raise ValueError("ATAC预测数据AnnData对象维度为零")
                
            return adata_omics3, adata_omics4
            
        except Exception as e:
            print(f"创建预测数据AnnData对象时出错: {str(e)}")
            # 使用原始数据创建替代对象
            print("使用原始数据作为后备方案")
            
            # 获取原始数据
            original_rna = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X.copy()
            original_atac = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else adata_omics2.X.copy()
            
            # 创建AnnData对象
            adata_omics3 = ad.AnnData(X=original_rna)
            adata_omics3.obs = adata_omics1.obs.copy()
            adata_omics3.var = adata_omics1.var.copy()
            if 'spatial' in adata_omics1.obsm:
                adata_omics3.obsm['spatial'] = adata_omics1.obsm['spatial'].copy()
                
            adata_omics4 = ad.AnnData(X=original_atac)
            adata_omics4.obs = adata_omics2.obs.copy()
            adata_omics4.var = adata_omics2.var.copy()
            if 'spatial' in adata_omics2.obsm:
                adata_omics4.obsm['spatial'] = adata_omics2.obsm['spatial'].copy()
            
            # 确保观测样本一致
            adata_omics4 = adata_omics4[adata_omics3.obs_names].copy()
            
            print(f"创建后备预测数据AnnData对象:")
            print(f"- RNA后备数据 (adata_omics3): {adata_omics3.shape}")
            print(f"- ATAC后备数据 (adata_omics4): {adata_omics4.shape}")
            
            return adata_omics3, adata_omics4
    
    def load_processed_data(self, adata_omics1=None, adata_omics2=None, save_prediction=True):
        """
        加载并预处理数据
        
        参数:
            adata_omics1: 可选，已加载的RNA数据AnnData对象
            adata_omics2: 可选，已加载的ATAC数据AnnData对象
            save_prediction: 是否将预测数据保存为独立的AnnData对象
            
        返回:
            如果save_prediction=False:
                处理后的adata_omics1, adata_omics2
            如果save_prediction=True:
                处理后的adata_omics1, adata_omics2, adata_omics3, adata_omics4
        """
        # 检查是否需要加载数据
        if adata_omics1 is None or adata_omics2 is None:
            print("需要提供RNA和ATAC数据")
            return None, None, None, None if save_prediction else None, None
        
        # 缓存文件路径
        rna_cache_file = os.path.join(self.cache_dir, f"processed_rna_{self.datatype}.h5ad")
        atac_cache_file = os.path.join(self.cache_dir, f"processed_atac_{self.datatype}.h5ad")
        rna_pred_file = os.path.join(self.cache_dir, f"adata_omics3.h5ad")
        atac_pred_file = os.path.join(self.cache_dir, f"adata_omics4.h5ad")
        
        # 如果不强制重新处理且缓存文件存在，直接加载缓存
        if not self.force_reprocess and os.path.exists(rna_cache_file) and os.path.exists(atac_cache_file):
            print(f"从h5ad缓存加载数据...")
            try:
                # 加载RNA缓存数据
                cached_rna = sc.read_h5ad(rna_cache_file)
                if 'original_expression' in cached_rna.obsm:
                    adata_omics1.obsm['original_expression'] = cached_rna.obsm['original_expression']
                if 'predicted_expression' in cached_rna.obsm:
                    adata_omics1.obsm['predicted_expression'] = cached_rna.obsm['predicted_expression']
                
                # 加载ATAC缓存数据
                cached_atac = sc.read_h5ad(atac_cache_file)
                if 'original_atac' in cached_atac.obsm:
                    adata_omics2.obsm['original_atac'] = cached_atac.obsm['original_atac']
                if 'predicted_atac' in cached_atac.obsm:
                    adata_omics2.obsm['predicted_atac'] = cached_atac.obsm['predicted_atac']
                
                print("缓存数据加载完成")
            except Exception as e:
                print(f"加载h5ad缓存出错: {str(e)}，将重新处理数据")
                self.force_reprocess = True
        
        # 如果需要重新处理数据
        if self.force_reprocess:
            print("处理RNA和ATAC数据...")
            
            # 使用GRN预测RNA数据
            adata_omics1 = self.predict_rna_from_grn(adata_omics1, adata_omics2)
            
            # 使用峰-TF映射预测ATAC数据
            adata_omics2 = self.predict_atac_from_peak_tf(adata_omics1, adata_omics2)
            
            # 保存缓存到h5ad文件
            print(f"保存处理后的数据到h5ad缓存文件")
            
            # 创建RNA数据的临时副本并保存
            rna_cache = adata_omics1.copy()
            print(f"- 保存RNA数据到: {rna_cache_file}")
            rna_cache.write_h5ad(rna_cache_file)
            
            # 创建ATAC数据的临时副本并保存
            atac_cache = adata_omics2.copy()
            print(f"- 保存ATAC数据到: {atac_cache_file}")
            atac_cache.write_h5ad(atac_cache_file)
        
        print("数据处理完成，原始数据和预测数据已分别存储在obsm中")
        print("- RNA原始数据: adata_omics1.obsm['original_expression']")
        print("- RNA预测数据: adata_omics1.obsm['predicted_expression']")
        print("- ATAC原始数据: adata_omics2.obsm['original_atac']")
        print("- ATAC预测数据: adata_omics2.obsm['predicted_atac']")
        
        # 如果需要保存预测数据为独立的AnnData对象
        if save_prediction:
            # 创建独立的预测数据AnnData对象
            adata_omics3, adata_omics4 = self.create_prediction_anndatas(adata_omics1, adata_omics2)
            
            # 保存预测数据AnnData对象
            print(f"保存预测数据为独立的AnnData对象")
            print(f"- 保存RNA预测数据到: {rna_pred_file}")
            adata_omics3.write_h5ad(rna_pred_file)
            
            print(f"- 保存ATAC预测数据到: {atac_pred_file}")
            adata_omics4.write_h5ad(atac_pred_file)
            
            return adata_omics1, adata_omics2, adata_omics3, adata_omics4
        
        return adata_omics1, adata_omics2

    def impute_rna_with_grn(self, adata_omics1, adata_omics2, confidence_threshold=0.3, min_cells=5, save_imputation_info=True):
        """
        使用GRN预测数据填补原始RNA数据矩阵，不修改原始数据
        
        参数:
            adata_omics1: RNA数据的AnnData对象（不会被修改）
            adata_omics2: ATAC数据的AnnData对象（不会被修改）
            confidence_threshold: 预测值的置信度阈值，低于此值的预测将被忽略
            min_cells: 基因在至少多少个细胞中表达才考虑填补
            save_imputation_info: 是否保存插补信息
            
        返回:
            adata_imputed_full: 插补后的完整RNA数据AnnData对象（原始数据+插补数据）
            adata_imputed_only: 只包含插补数据的AnnData对象
            imputation_info: 包含插补信息的字典
        """
        print("使用GRN预测数据填补RNA数据（不修改原始数据）...")
        
        # 确保TF-基因网络已加载（使用上下文基因过滤）
        self._ensure_tf_gene_network(adata_omics1)
        
        if self.tf_gene_network is None:
            print("警告: 未加载TF-基因调控网络，无法进行数据填补")
            # 创建原始数据的副本作为返回
            adata_imputed_full = adata_omics1.copy()
            empty_imputed = adata_omics1.copy()
            empty_imputed.X = scipy.sparse.csr_matrix(empty_imputed.X.shape)
            empty_info = {
                'filled_genes': 0,
                'filled_values': 0,
                'imputed_gene_names': [],
                'imputation_summary': {}
            }
            return adata_imputed_full, empty_imputed, empty_info
        
        # 创建原始数据的副本，避免修改原始数据
        adata_imputed_full = adata_omics1.copy()
        adata_imputed_only = adata_omics1.copy()
        
        # 获取表达矩阵副本
        X_orig = adata_imputed_full.X.toarray() if scipy.sparse.issparse(adata_imputed_full.X) else adata_imputed_full.X.copy()
        X_original_backup = X_orig.copy()  # 保存原始数据的完整副本
        X_imputed_only = np.zeros_like(X_orig)  # 创建全零矩阵用于存储插补值
        
        # 初始化插补跟踪矩阵
        imputation_mask = np.zeros_like(X_orig, dtype=bool)  # 标记哪些位置被插补了
        imputation_values = np.zeros_like(X_orig)  # 记录插补的值
        
        # 创建基因到索引的映射
        gene_to_idx = {gene: i for i, gene in enumerate(adata_imputed_full.var_names)}
        
        # 找出所有可预测的基因（在GRN网络中的基因）
        predictable_genes = set()
        for tf in sorted(self.tf_gene_network.keys()):
            targets = self.tf_gene_network[tf]
            if tf in gene_to_idx:  # 确保TF在数据集中
                for target in targets:
                    if target in gene_to_idx:  # 确保目标基因在数据集中
                        predictable_genes.add(target)
        
        print(f"在GRN网络中找到{len(predictable_genes)}个可预测的基因")
        
        # 计算每个基因的表达频率
        gene_expression_freq = np.sum(X_orig > 0, axis=0)
        
        # 找出需要填补的基因
        genes_to_impute = []
        for gene in sorted(predictable_genes):
            gene_idx = gene_to_idx[gene]
            if gene_expression_freq[gene_idx] < min_cells:
                genes_to_impute.append(gene_idx)
        
        print(f"找到{len(genes_to_impute)}个需要填补的基因")
        
        # 对每个需要填补的基因进行处理
        filled_genes = 0
        filled_values = 0
        imputed_gene_names = []
        imputation_summary = {}
        
        for gene_idx in tqdm(genes_to_impute, desc="填补基因表达"):
            gene_name = adata_imputed_full.var_names[gene_idx]
            
            # 找出调控该基因的TF
            regulating_tfs = []
            for tf in sorted(self.tf_gene_network.keys()):
                targets = self.tf_gene_network[tf]
                if gene_name in targets and tf in gene_to_idx:
                    regulating_tfs.append(tf)
            
            if not regulating_tfs:
                continue
            
            # 计算TF表达的平均值作为预测值
            tf_expr_sum = np.zeros(X_orig.shape[0])
            for tf in sorted(regulating_tfs):
                tf_idx = gene_to_idx[tf]
                tf_expr_sum += X_orig[:, tf_idx]
            
            # 计算平均TF表达
            predicted_expr = tf_expr_sum / len(regulating_tfs)
            
            # 应用置信度阈值
            predicted_expr[predicted_expr < confidence_threshold] = 0
            
            # 只填补原始数据中为零的位置
            original_zero_mask = X_orig[:, gene_idx] == 0
            valid_imputation_mask = original_zero_mask & (predicted_expr > 0)
            changes = np.sum(valid_imputation_mask)
            
            if changes > 0:
                # 更新插补后的完整数据
                X_orig[valid_imputation_mask, gene_idx] = predicted_expr[valid_imputation_mask]
                
                # 更新只包含插补值的数据
                X_imputed_only[valid_imputation_mask, gene_idx] = predicted_expr[valid_imputation_mask]
                
                # 记录插补信息
                imputation_mask[valid_imputation_mask, gene_idx] = True
                imputation_values[valid_imputation_mask, gene_idx] = predicted_expr[valid_imputation_mask]
                
                # 统计信息
                filled_genes += 1
                filled_values += changes
                imputed_gene_names.append(gene_name)
                
                # 详细的插补统计
                imputation_summary[gene_name] = {
                    'gene_index': gene_idx,
                    'regulating_tfs': sorted(regulating_tfs),
                    'num_tfs': len(regulating_tfs),
                    'imputed_positions': changes,
                    'original_zeros': np.sum(original_zero_mask),
                    'imputation_rate': changes / np.sum(original_zero_mask) if np.sum(original_zero_mask) > 0 else 0,
                    'avg_imputed_value': np.mean(predicted_expr[valid_imputation_mask]),
                    'min_imputed_value': np.min(predicted_expr[valid_imputation_mask]),
                    'max_imputed_value': np.max(predicted_expr[valid_imputation_mask]),
                    'imputed_cell_indices': np.where(valid_imputation_mask)[0].tolist()
                }
                
                #print(f"  {gene_name}: 插补了{changes}个位置，平均值={np.mean(predicted_expr[valid_imputation_mask]):.4f}")
        
        # 更新AnnData对象的表达矩阵
        adata_imputed_full.X = scipy.sparse.csr_matrix(X_orig)
        adata_imputed_only.X = scipy.sparse.csr_matrix(X_imputed_only)
        
        # 保存插补信息
        if save_imputation_info:
            # 保存到插补后完整数据对象中
            adata_imputed_full.layers['original_data'] = scipy.sparse.csr_matrix(X_original_backup)
            adata_imputed_full.layers['imputation_mask'] = scipy.sparse.csr_matrix(imputation_mask.astype(float))
            adata_imputed_full.layers['imputation_values'] = scipy.sparse.csr_matrix(imputation_values)
            adata_imputed_full.layers['imputed_only'] = adata_imputed_only.X.copy()
            
            # 保存到只包含插补数据的对象中
            adata_imputed_only.layers['original_data'] = scipy.sparse.csr_matrix(X_original_backup)
            adata_imputed_only.layers['imputed_full'] = adata_imputed_full.X.copy()
            adata_imputed_only.layers['imputation_mask'] = scipy.sparse.csr_matrix(imputation_mask.astype(float))
            adata_imputed_only.layers['imputation_values'] = scipy.sparse.csr_matrix(imputation_values)
            
            # 在var中添加插补标记和统计信息
            for adata in [adata_imputed_full, adata_imputed_only]:
                adata.var['was_imputed'] = False
                for gene_name in imputed_gene_names:
                    adata.var.loc[gene_name, 'was_imputed'] = True
                
                # 添加插补统计信息
                imputation_counts = np.zeros(adata.n_vars)
                for gene_name in imputed_gene_names:
                    gene_idx = gene_to_idx[gene_name]
                    imputation_counts[gene_idx] = imputation_summary[gene_name]['imputed_positions']
                adata.var['imputation_count'] = imputation_counts
                
                # 保存插补详细信息到uns
                adata.uns['imputation_info'] = {
                    'method': 'GRN_based',
                    'confidence_threshold': confidence_threshold,
                    'min_cells': min_cells,
                    'total_filled_genes': filled_genes,
                    'total_filled_values': filled_values,
                    'imputation_summary': imputation_summary
                }
            
            # 添加数据类型标记
            adata_imputed_full.uns['data_type'] = 'original_plus_imputed'
            adata_imputed_full.uns['description'] = 'Contains original data plus imputed values'
            
            adata_imputed_only.uns['data_type'] = 'imputed_only'
            adata_imputed_only.uns['description'] = 'Contains only imputed values, non-imputed positions are zero'
            
            print("插补信息已保存到AnnData对象中:")
            print("插补后完整数据 (adata_imputed_full):")
            print("- .X: 原始数据 + 插补数据")
            print("- .layers['original_data']: 原始未插补数据")
            print("- .layers['imputation_mask']: 插补位置掩码")
            print("- .layers['imputation_values']: 插补的值")
            print("- .layers['imputed_only']: 只包含插补值的数据")
            print("只包含插补数据 (adata_imputed_only):")
            print("- .X: 只包含插补值（其他位置为0）")
            print("- .layers['original_data']: 原始未插补数据")
            print("- .layers['imputed_full']: 完整的插补后数据")
            print("- .layers['imputation_mask']: 插补位置掩码")
        
        # 在obs中添加每个细胞的插补统计
        imputed_per_cell = np.sum(imputation_mask, axis=1)
        for adata in [adata_imputed_full, adata_imputed_only]:
            adata.obs['imputation_count'] = imputed_per_cell
            adata.obs['has_imputation'] = imputed_per_cell > 0
            adata.obs['imputation_rate'] = imputed_per_cell / adata.n_vars
        
        # 计算填补效果
        print(f"\n填补效果统计:")
        print(f"- 成功填补了{filled_genes}个基因")
        print(f"- 总共填补了{filled_values}个值")
        print(f"- 填补前非零元素比例: {np.count_nonzero(X_original_backup) / X_original_backup.size:.2%}")
        print(f"- 填补后非零元素比例: {np.count_nonzero(X_orig) / X_orig.size:.2%}")
        print(f"- 插补数据非零元素比例: {np.count_nonzero(X_imputed_only) / X_imputed_only.size:.2%}")
        print(f"- 插补率: {filled_values / X_orig.size:.4%}")
        
        # 验证原始数据未被修改
        original_unchanged = np.array_equal(
            adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X,
            X_original_backup
        )
        print(f"- 原始数据未被修改: {original_unchanged}")
        
        # 创建返回的插补信息字典
        imputation_info = {
            'filled_genes': filled_genes,
            'filled_values': filled_values,
            'imputed_gene_names': imputed_gene_names,
            'imputation_summary': imputation_summary,
            'confidence_threshold': confidence_threshold,
            'min_cells': min_cells,
            'imputation_rate': filled_values / X_orig.size,
            'genes_to_impute': [adata_imputed_full.var_names[idx] for idx in genes_to_impute],
            'imputation_mask': imputation_mask,
            'original_data_shape': X_original_backup.shape,
            'imputed_data_shape': X_imputed_only.shape
        }
        
        return adata_imputed_full, adata_imputed_only, imputation_info

    def evaluate_imputation_quality(self, X_orig, X_imputed):
        """评估填补质量"""
        # 计算填补前后的非零比例
        original_nonzero = np.count_nonzero(X_orig) / X_orig.size
        imputed_nonzero = np.count_nonzero(X_imputed) / X_imputed.size
        
        # 计算填补前后的基因表达分布
        original_means = np.mean(X_orig, axis=0)
        imputed_means = np.mean(X_imputed, axis=0)
        
        # 计算填补前后的相关性
        correlation = np.corrcoef(original_means, imputed_means)[0,1]
        
        print(f"填补效果统计:")
        print(f"- 非零比例: {original_nonzero:.2%} -> {imputed_nonzero:.2%}")
        print(f"- 基因表达相关性: {correlation:.4f}")

    def impute_atac_with_peak_tf(self, adata_omics1, adata_omics2, confidence_threshold=0.3, min_cells=5, save_imputation_info=True):
        """
        使用峰-TF映射关系填补ATAC数据，不修改原始数据
        
        参数:
            adata_omics1: RNA数据的AnnData对象（不会被修改）
            adata_omics2: ATAC数据的AnnData对象（不会被修改）
            confidence_threshold: 预测值的置信度阈值
            min_cells: 峰在至少多少个细胞中开放才考虑填补
            save_imputation_info: 是否保存插补信息
            
        返回:
            adata_imputed_full: 插补后的完整ATAC数据AnnData对象（原始数据+插补数据）
            adata_imputed_only: 只包含插补数据的AnnData对象
            imputation_info: 包含插补信息的字典
        """
        print("使用峰-TF映射关系填补ATAC数据（不修改原始数据）...")
        
        if self.peak_tf_map is None:
            print("警告: 未加载峰-TF映射关系，无法进行数据填补")
            # 创建原始数据的副本作为返回
            adata_imputed_full = adata_omics2.copy()
            empty_imputed = adata_omics2.copy()
            empty_imputed.X = scipy.sparse.csr_matrix(empty_imputed.X.shape)
            empty_info = {
                'filled_peaks': 0,
                'filled_values': 0,
                'imputed_peak_names': [],
                'imputation_summary': {}
            }
            return adata_imputed_full, empty_imputed, empty_info
        
        # 创建原始数据的副本，避免修改原始数据
        adata_imputed_full = adata_omics2.copy()
        adata_imputed_only = adata_omics2.copy()
        
        # 获取数据矩阵副本
        X_atac = adata_imputed_full.X.toarray() if scipy.sparse.issparse(adata_imputed_full.X) else adata_imputed_full.X.copy()
        X_atac_backup = X_atac.copy()  # 保存原始数据
        X_imputed_only = np.zeros_like(X_atac)  # 创建全零矩阵
        
        # 获取RNA表达矩阵
        X_rna = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X
        
        # 初始化插补跟踪矩阵
        imputation_mask = np.zeros_like(X_atac, dtype=bool)
        imputation_values = np.zeros_like(X_atac)
        
        # 创建基因到索引的映射
        gene_to_idx = {gene: i for i, gene in enumerate(adata_omics1.var_names)}
        
        # 创建峰到索引的映射
        peak_to_idx = {}
        for i, peak in enumerate(adata_imputed_full.var_names):
            peak_to_idx[peak] = i
            
            # 处理可能的格式差异
            if ':' in peak:
                parts = peak.split(':')
                if len(parts) == 2 and '-' in parts[1]:
                    chrom = parts[0]
                    pos = parts[1].split('-')
                    if len(pos) == 2:
                        alt_peak = f"{chrom}-{pos[0]}-{pos[1]}"
                        peak_to_idx[alt_peak] = i
            elif '-' in peak:
                parts = peak.split('-')
                if len(parts) >= 3:
                    chrom, start, end = parts[0], parts[1], parts[2]
                    alt_peak = f"{chrom}:{start}-{end}"
                    peak_to_idx[alt_peak] = i
        
        # 找出可预测的峰
        predictable_peaks_idx = set()
        peak_tf_matches = 0
        
        for peak in sorted(self.peak_tf_map.keys()):
            if peak in peak_to_idx:
                peak_tf_matches += 1
                predictable_peaks_idx.add(peak_to_idx[peak])
        
        print(f"在峰-TF映射中找到{peak_tf_matches}个匹配的峰")
        
        # 统计信息
        filled_peaks = 0
        filled_values = 0
        imputed_peak_names = []
        imputation_summary = {}
        
        # 对可预测的峰进行填补
        for peak_idx in tqdm(sorted(predictable_peaks_idx), desc="填补峰开放状态"):
            peak_name = adata_imputed_full.var_names[peak_idx]
            
            # 查找调控该峰的TF
            regulating_tfs = []
            for peak in sorted(self.peak_tf_map.keys()):
                tfs = self.peak_tf_map[peak]
                if peak in peak_to_idx and peak_to_idx[peak] == peak_idx:
                    regulating_tfs = tfs
                    break
            
            if not regulating_tfs:
                continue
            
            # 找到在RNA数据中的TF
            valid_tfs = sorted([tf for tf in regulating_tfs if tf in gene_to_idx])
            if not valid_tfs:
                continue
            
            # 获取零值细胞位置
            original_zero_mask = X_atac[:, peak_idx] == 0
            if np.sum(original_zero_mask) == 0:
                continue
            
            # 计算TF表达的平均值作为预测值
            tf_expr_values = np.array([X_rna[:, gene_to_idx[tf]] for tf in valid_tfs])
            predicted_expr = np.mean(tf_expr_values, axis=0)
            
            # 应用置信度阈值
            predicted_expr[predicted_expr < confidence_threshold] = 0
            
            # 只填补原始数据中为零的位置
            valid_imputation_mask = original_zero_mask & (predicted_expr > 0)
            changes = np.sum(valid_imputation_mask)
            
            if changes > 0:
                # 更新插补后的完整数据
                X_atac[valid_imputation_mask, peak_idx] = predicted_expr[valid_imputation_mask]
                
                # 更新只包含插补值的数据
                X_imputed_only[valid_imputation_mask, peak_idx] = predicted_expr[valid_imputation_mask]
                
                # 记录插补信息
                imputation_mask[valid_imputation_mask, peak_idx] = True
                imputation_values[valid_imputation_mask, peak_idx] = predicted_expr[valid_imputation_mask]
                
                # 统计信息
                filled_peaks += 1
                filled_values += changes
                imputed_peak_names.append(peak_name)
                
                # 详细的插补统计
                imputation_summary[peak_name] = {
                    'peak_index': peak_idx,
                    'regulating_tfs': sorted(valid_tfs),
                    'num_tfs': len(valid_tfs),
                    'imputed_positions': changes,
                    'original_zeros': np.sum(original_zero_mask),
                    'imputation_rate': changes / np.sum(original_zero_mask) if np.sum(original_zero_mask) > 0 else 0,
                    'avg_imputed_value': np.mean(predicted_expr[valid_imputation_mask]),
                    'min_imputed_value': np.min(predicted_expr[valid_imputation_mask]),
                    'max_imputed_value': np.max(predicted_expr[valid_imputation_mask]),
                    'imputed_cell_indices': np.where(valid_imputation_mask)[0].tolist()
                }
        
        # 更新AnnData对象的表达矩阵
        adata_imputed_full.X = scipy.sparse.csr_matrix(X_atac)
        adata_imputed_only.X = scipy.sparse.csr_matrix(X_imputed_only)
        
        # 保存插补信息
        if save_imputation_info:
            # 保存到插补后完整数据对象中
            adata_imputed_full.layers['original_data'] = scipy.sparse.csr_matrix(X_atac_backup)
            adata_imputed_full.layers['imputation_mask'] = scipy.sparse.csr_matrix(imputation_mask.astype(float))
            adata_imputed_full.layers['imputation_values'] = scipy.sparse.csr_matrix(imputation_values)
            adata_imputed_full.layers['imputed_only'] = adata_imputed_only.X.copy()
            
            # 保存到只包含插补数据的对象中
            adata_imputed_only.layers['original_data'] = scipy.sparse.csr_matrix(X_atac_backup)
            adata_imputed_only.layers['imputed_full'] = adata_imputed_full.X.copy()
            adata_imputed_only.layers['imputation_mask'] = scipy.sparse.csr_matrix(imputation_mask.astype(float))
            adata_imputed_only.layers['imputation_values'] = scipy.sparse.csr_matrix(imputation_values)
            
            # 添加元数据标记和统计信息
            for adata in [adata_imputed_full, adata_imputed_only]:
                adata.var['was_imputed'] = False
                for peak_name in imputed_peak_names:
                    if peak_name in adata.var_names:
                        adata.var.loc[peak_name, 'was_imputed'] = True
                
                # 添加插补统计信息
                imputation_counts = np.zeros(adata.n_vars)
                for peak_name in imputed_peak_names:
                    if peak_name in adata.var_names and peak_name in imputation_summary:
                        peak_idx = list(adata.var_names).index(peak_name)
                        imputation_counts[peak_idx] = imputation_summary[peak_name]['imputed_positions']
                adata.var['imputation_count'] = imputation_counts
                
                # 保存详细信息到uns
                adata.uns['imputation_info'] = {
                    'method': 'PeakTF_based',
                    'confidence_threshold': confidence_threshold,
                    'min_cells': min_cells,
                    'total_filled_peaks': filled_peaks,
                    'total_filled_values': filled_values,
                    'imputation_summary': imputation_summary
                }
            
            # 添加数据类型标记
            adata_imputed_full.uns['data_type'] = 'original_plus_imputed'
            adata_imputed_only.uns['data_type'] = 'imputed_only'
        
        # 在obs中添加每个细胞的插补统计
        imputed_per_cell = np.sum(imputation_mask, axis=1)
        for adata in [adata_imputed_full, adata_imputed_only]:
            adata.obs['imputation_count'] = imputed_per_cell
            adata.obs['has_imputation'] = imputed_per_cell > 0
            adata.obs['imputation_rate'] = imputed_per_cell / adata.n_vars
        
        # 计算填补效果
        print(f"\n填补效果统计:")
        print(f"- 成功填补了{filled_peaks}个峰")
        print(f"- 总共填补了{filled_values}个值")
        print(f"- 填补前非零元素比例: {np.count_nonzero(X_atac_backup) / X_atac_backup.size:.2%}")
        print(f"- 填补后非零元素比例: {np.count_nonzero(X_atac) / X_atac.size:.2%}")
        print(f"- 插补数据非零元素比例: {np.count_nonzero(X_imputed_only) / X_imputed_only.size:.2%}")
        
        # 验证原始数据未被修改
        original_unchanged = np.array_equal(
            adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else adata_omics2.X,
            X_atac_backup
        )
        print(f"- 原始数据未被修改: {original_unchanged}")
        
        # 创建返回的插补信息字典
        imputation_info = {
            'filled_peaks': filled_peaks,
            'filled_values': filled_values,
            'imputed_peak_names': imputed_peak_names,
            'imputation_summary': imputation_summary,
            'confidence_threshold': confidence_threshold,
            'min_cells': min_cells,
            'imputation_rate': filled_values / X_atac.size,
            'imputation_mask': imputation_mask,
            'original_data_shape': X_atac_backup.shape,
            'imputed_data_shape': X_imputed_only.shape
        }
        
        return adata_imputed_full, adata_imputed_only, imputation_info

    def load_ppi_network(self):
        """加载STRING格式的PPI网络文件，优先从缓存加载，集成蛋白质ID映射器"""
        
        # 初始化必要的属性（如果尚未初始化）
        if not hasattr(self, 'ppi_network_cache'):
            self.ppi_network_cache = os.path.join(self.cache_dir, 'ppi_network.pkl')
        if not hasattr(self, 'protein_mapping_cache'):
            self.protein_mapping_cache = os.path.join(self.cache_dir, 'protein_mapping.pkl')
        if not hasattr(self, 'ppi_network'):
            self.ppi_network = None
        
        # 初始化蛋白质ID映射器
        if not hasattr(self, 'protein_id_mapper') or self.protein_id_mapper is None:
            self._initialize_protein_id_mapper()
            
        # 如果PPI网络已加载，直接返回
        if self.ppi_network is not None:
            print(f"使用已加载的PPI网络")
            return self.ppi_network
        
        # 尝试从缓存加载
        if not self.force_reprocess and os.path.exists(self.ppi_network_cache):
            try:
                print(f"从缓存加载PPI网络: {self.ppi_network_cache}")
                import pickle
                with open(self.ppi_network_cache, 'rb') as f:
                    self.ppi_network = pickle.load(f)
                
                # 验证加载的网络
                if self.ppi_network and isinstance(self.ppi_network, dict):
                    num_proteins = len(self.ppi_network)
                    num_interactions = sum(len(interactions) for interactions in self.ppi_network.values()) // 2
                    print(f"成功从缓存加载PPI网络: {num_proteins} 个蛋白质, {num_interactions} 个互作")
                    return self.ppi_network
                else:
                    print("缓存中的PPI网络无效，将重新加载")
                    self.ppi_network = None
            except Exception as e:
                print(f"加载PPI网络缓存失败: {str(e)}")
                self.ppi_network = None
        
        # 如果缓存加载失败或强制重新处理，从文件加载
        print(f"从文件加载PPI网络: {self.ppi_file_path}")
        
        try:
            # 首先检查PPI文件路径是否有效
            if self.ppi_file_path is None:
                print(f"错误: PPI文件路径未设置")
                return None
                
            # 检查文件是否存在
            if not os.path.exists(self.ppi_file_path):
                print(f"错误: PPI文件不存在: {self.ppi_file_path}")
                return None
                
            # 读取文件前几行来确定格式
            with open(self.ppi_file_path, 'r') as f:
                first_lines = [next(f) for _ in range(5) if f]
                
            # 检查文件格式
            if len(first_lines) == 0:
                print(f"错误: PPI文件为空")
                return None
                
            # 判断分隔符
            delimiter = '\t' if '\t' in first_lines[0] else ' '
            print(f"使用分隔符: '{delimiter}'")
            
            # 检查是否有标题行
            has_header = False
            if any(x in first_lines[0].lower() for x in ['protein', 'score', 'combined']):
                has_header = True
                print("检测到标题行")
            
            # 获取ADT数据相关的蛋白质ID
            relevant_protein_ids = set()
            if self.protein_id_mapper:
                print("使用蛋白质ID映射器获取相关的STRING ID...")
                
                # 1. 从手动映射中获取所有STRING ID（现在为空，主要依赖STRING数据库）
                manual_count = 0
                for protein_name, mapping_data in self.protein_id_mapper.manual_mapping.items():
                    if isinstance(mapping_data, list) and len(mapping_data) >= 2:
                        # 兼容原格式: [基因ID, [蛋白质ID列表]]
                        protein_ids = mapping_data[1] if isinstance(mapping_data[1], list) else [mapping_data[1]]
                        for protein_id in protein_ids:
                            relevant_protein_ids.add(protein_id)
                            manual_count += 1
                print(f"从手动映射获得 {manual_count} 个蛋白质ID")
                
                # 2. 从基因符号映射中获取
                if hasattr(self.protein_id_mapper, 'gene_symbol_to_string_ids'):
                    gene_symbol_count = 0
                    for string_ids in self.protein_id_mapper.gene_symbol_to_string_ids.values():
                        relevant_protein_ids.update(string_ids)
                        gene_symbol_count += len(string_ids)
                    print(f"从基因符号映射获得 {gene_symbol_count} 个STRING ID")
                
                # 3. 从首选名称映射中获取
                if hasattr(self.protein_id_mapper, 'preferred_name_to_string_id'):
                    preferred_count = 0
                    for string_ids in self.protein_id_mapper.preferred_name_to_string_id.values():
                        relevant_protein_ids.update(string_ids)
                        preferred_count += len(string_ids)
                    print(f"从首选名称映射获得 {preferred_count} 个STRING ID")
                
                print(f"从映射器总共获得 {len(relevant_protein_ids)} 个唯一的蛋白质ID")
            else:
                print("警告: 蛋白质ID映射器未初始化")
            
            # 如果没有找到相关蛋白质ID，使用测试ID
            if not relevant_protein_ids:
                test_ids = [
                    'ENSP00000359722',  # CD163
                    'ENSP00000367890',  # CR2
                    'ENSP00000368438',  # PCNA
                    'ENSP00000224237',  # VIM
                    'ENSP00000252242',  # KRT5
                    'ENSP00000358022',  # CD68
                ]
                relevant_protein_ids.update(test_ids)
            
            print(f"找到 {len(relevant_protein_ids)} 个与ADT数据相关的蛋白质ID")
            
            # 读取STRING格式的PPI数据，只加载与相关蛋白质相互作用的数据
            print("开始读取PPI数据...")
            
            # 使用分块读取以减少内存使用
            chunk_size = 100000  # 每次读取的行数
            ppi_network = {}
            total_lines = 0
            relevant_lines = 0
            
            # 确定列名
            if has_header:
                df_header = pd.read_csv(self.ppi_file_path, sep=delimiter, nrows=0)
                protein1_col = next((col for col in df_header.columns if 'protein1' in str(col).lower()), None)
                protein2_col = next((col for col in df_header.columns if 'protein2' in str(col).lower()), None)
                score_col = next((col for col in df_header.columns if 'score' in str(col).lower() or 'combined' in str(col).lower()), None)
                
                if not (protein1_col and protein2_col and score_col):
                    print(f"警告: 无法识别列名, 列名: {list(df_header.columns)}")
                    protein1_col = df_header.columns[0]
                    protein2_col = df_header.columns[1]
                    score_col = df_header.columns[2]
            else:
                # 假设前三列分别是protein1, protein2, score
                protein1_col = 0
                protein2_col = 1
                score_col = 2
            
            print(f"使用列: protein1={protein1_col}, protein2={protein2_col}, score={score_col}")
            
            # 分块读取文件
            for chunk in pd.read_csv(self.ppi_file_path, sep=delimiter, chunksize=chunk_size, header=0 if has_header else None):
                total_lines += len(chunk)
                
                # 过滤只包含相关蛋白质的行
                if isinstance(protein1_col, int):
                    # 使用列索引
                    relevant_chunk = chunk[(chunk.iloc[:, protein1_col].isin(relevant_protein_ids)) | 
                                         (chunk.iloc[:, protein2_col].isin(relevant_protein_ids))]
                else:
                    # 使用列名
                    relevant_chunk = chunk[(chunk[protein1_col].isin(relevant_protein_ids)) | 
                                         (chunk[protein2_col].isin(relevant_protein_ids))]
                
                relevant_lines += len(relevant_chunk)
                
                # 处理相关行
                for _, row in relevant_chunk.iterrows():
                    p1 = str(row[protein1_col])
                    p2 = str(row[protein2_col])
                    
                    # 确保蛋白质ID在网络中有条目
                    if p1 not in ppi_network:
                        ppi_network[p1] = {}
                    if p2 not in ppi_network:
                        ppi_network[p2] = {}
                    
                    # 确保分数是数值
                    try:
                        conf = float(row[score_col])
                        # 如果分数范围是0-1000，归一化到0-1
                        if conf > 1:
                            conf = conf / 1000.0
                    except:
                        # 如果转换失败，使用默认值
                        conf = 0.5
                    
                    # 添加双向关系
                    ppi_network[p1][p2] = conf
                    ppi_network[p2][p1] = conf
                
                # 打印进度
                if total_lines % (chunk_size * 10) == 0:
                    print(f"已处理 {total_lines} 行, 找到 {relevant_lines} 行相关数据")
            
            # 检查网络构建结果
            if not ppi_network:
                print("错误: 未能构建PPI网络")
                return None
                
            # 统计网络信息
            num_proteins = len(ppi_network)
            num_interactions = sum(len(interactions) for interactions in ppi_network.values()) // 2  # 除以2因为每个相互作用被计算了两次
            
            print(f"成功加载PPI网络: {num_proteins} 个蛋白质, {num_interactions} 个互作")
            print(f"从原始数据 {total_lines} 行中筛选出 {relevant_lines} 行相关数据")
            
            # 检查我们的手动映射ID是否在网络中
            test_ids = [
                'ENSP00000359722',  # CD163
                'ENSP00000367890',  # CR2
                'ENSP00000368438'   # PCNA
            ]
            
            print("\n检查手动映射ID是否在PPI网络中:")
            for test_id in test_ids:
                if test_id in ppi_network:
                    num_interactions = len(ppi_network[test_id])
                    print(f"  {test_id} 在网络中，有 {num_interactions} 个互作")
                    # 打印前3个互作伙伴
                    partners = list(ppi_network[test_id].keys())[:3]
                    print(f"    互作伙伴示例: {partners}")
                else:
                    print(f"  {test_id} 不在网络中")
            
            # 保存PPI网络到缓存
            try:
                print(f"保存PPI网络到缓存: {self.ppi_network_cache}")
                import pickle
                with open(self.ppi_network_cache, 'wb') as f:
                    pickle.dump(ppi_network, f)
                print("PPI网络已保存到缓存")
            except Exception as e:
                print(f"保存PPI网络缓存失败: {str(e)}")
            
            # 保存加载的网络到实例变量
            self.ppi_network = ppi_network
            return ppi_network
        except Exception as e:
            print(f"加载PPI网络出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def impute_protein_with_ppi(self, adata, confidence_threshold=0.7, use_rna_data=False, rna_adata=None, max_impute_rate=0.05, min_confidence_for_imputation=0.8, top_percent_to_impute=10):
        """
        使用PPI网络填补蛋白质表达数据
        
        参数:
            adata: 包含蛋白质表达数据的AnnData对象
            confidence_threshold: PPI置信度阈值，只使用高于此值的相互作用
            use_rna_data: 是否使用RNA数据进行填补
            rna_adata: RNA数据的AnnData对象，当use_rna_data=True时使用
            max_impute_rate: 最大填补率，默认为0.05（5%）
            min_confidence_for_imputation: 填补时的最低置信度阈值，默认为0.8
            top_percent_to_impute: 只填补预测值在前N%的位置，默认为10%
            
        返回:
            更新后的AnnData对象
        """
        # 确保蛋白质ID映射器已初始化
        if not hasattr(self, 'protein_id_mapper') or self.protein_id_mapper is None:
            self._initialize_protein_id_mapper()
            
        print("使用PPI网络填补蛋白质表达数据...")
        print(f"使用置信度阈值: {confidence_threshold}")
        print(f"最大填补率: {max_impute_rate*100}%")
        print(f"填补最低置信度阈值: {min_confidence_for_imputation}")
        print(f"只填补预测值前{top_percent_to_impute}%的位置")
        
        # 加载PPI网络
        ppi_network = self.load_ppi_network()
        if ppi_network is None:
            print("警告: 未加载PPI网络，无法进行数据填补")
            return adata
        
        # 获取原始表达矩阵
        X_orig = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
        
        # 创建蛋白质到索引的映射
        protein_to_idx = {protein: i for i, protein in enumerate(adata.var_names)}
        print(f"ADT数据中的蛋白质数量: {len(protein_to_idx)}")
        print(f"前10个蛋白质名称: {list(adata.var_names[:10])}")
        
        # 检查是否已有STRING ID映射
        has_string_ids = 'string_ids' in adata.var.columns
        if has_string_ids:
            print("检测到已有STRING ID映射，将使用现有映射")
            
        # 创建STRING ID到ADT蛋白质的映射
        string_to_adt = {}
        adt_found_in_ppi = set()
        
        if has_string_ids:
            # 使用已有的STRING ID映射
            for i, protein in enumerate(adata.var_names):
                string_ids_str = adata.var['string_ids'].iloc[i]  # 使用.iloc避免FutureWarning
                if string_ids_str:  # 如果有映射
                    # 分割字符串获取所有STRING ID
                    string_ids = string_ids_str.split(',')
                    
                    # 检查每个STRING ID是否在PPI网络中
                    for string_id in string_ids:
                        if string_id in ppi_network:
                            string_to_adt[string_id] = protein
                            adt_found_in_ppi.add(protein)
            
            print(f"使用已有映射找到的蛋白质: {len(adt_found_in_ppi)}")
            if len(adt_found_in_ppi) > 0:
                print(f"已映射蛋白质示例: {list(adt_found_in_ppi)[:5]}")
                
            # 打印前10个STRING ID及其是否在PPI网络中
            print("\n检查前10个蛋白质的STRING ID是否在PPI网络中:")
            for i in range(min(10, adata.n_vars)):
                protein = adata.var_names[i]
                string_ids_str = adata.var['string_ids'].iloc[i]
                if string_ids_str:
                    string_ids = string_ids_str.split(',')
                    
                    in_network = [id for id in string_ids if id in ppi_network]
                    print(f"  {protein}: STRING IDs={string_ids}, 在网络中={in_network}")
                else:
                    print(f"  {protein}: 无STRING ID映射")
        
        # 如果没有找到足够的蛋白质，尝试其他匹配方法
        if len(adt_found_in_ppi) < len(protein_to_idx) * 0.1:  # 如果匹配率低于10%
            print("已有映射匹配率太低，尝试其他匹配方法...")
            
            # 1. 直接匹配
            direct_matched = set()
            for protein in adata.var_names:
                if protein not in adt_found_in_ppi:
                    if protein in ppi_network:
                        string_to_adt[protein] = protein
                        adt_found_in_ppi.add(protein)
                        direct_matched.add(protein)
            
            print(f"直接匹配到的蛋白质: {len(direct_matched)}")
            if len(direct_matched) > 0:
                print(f"直接匹配的蛋白质示例: {list(direct_matched)[:5]}")
            
            # 2. 尝试去掉前缀后匹配
            prefix_matched = set()
            for protein in adata.var_names:
                if protein not in adt_found_in_ppi:
                    # 如果蛋白质名称包含下划线或点，尝试分割
                    parts = protein.split('_')
                    if len(parts) > 1:
                        base_name = parts[-1]  # 取最后一部分
                        if base_name in ppi_network:
                            string_to_adt[base_name] = protein
                            adt_found_in_ppi.add(protein)
                            prefix_matched.add(protein)
            
            print(f"去掉前缀后匹配到的蛋白质: {len(prefix_matched)}")
            if len(prefix_matched) > 0:
                print(f"前缀匹配的蛋白质示例: {list(prefix_matched)[:5]}")
            
            # 3. 模糊匹配 - 检查STRING ID是否包含ADT蛋白质名称
            fuzzy_matched = set()
            for protein in adata.var_names:
                if protein not in adt_found_in_ppi:
                    for string_id in ppi_network.keys():
                        # 检查STRING ID是否包含蛋白质名称
                        if protein.lower() in string_id.lower():
                            string_to_adt[string_id] = protein
                            adt_found_in_ppi.add(protein)
                            fuzzy_matched.add(protein)
                            break
            
            print(f"模糊匹配到的蛋白质数量: {len(fuzzy_matched)}")
            if len(fuzzy_matched) > 0:
                print(f"模糊匹配的蛋白质示例: {list(fuzzy_matched)[:5]}")
            
            # 5. 使用手动映射
            manual_matched = set()
            if hasattr(self, 'protein_id_mapper') and self.protein_id_mapper:
                print("使用蛋白质ID映射器进行匹配...")
                for protein in adata.var_names:
                    if protein not in adt_found_in_ppi:
                        # 使用映射器获取STRING ID
                        string_ids = self.protein_id_mapper.map_protein_to_string_ids(protein)
                        for string_id in string_ids:
                            if string_id in ppi_network:
                                string_to_adt[string_id] = protein
                                adt_found_in_ppi.add(protein)
                                manual_matched.add(protein)
                                break
                
                print(f"使用映射器匹配到的蛋白质: {len(manual_matched)}")
                if len(manual_matched) > 0:
                    print(f"映射器匹配的蛋白质示例: {list(manual_matched)[:5]}")
            
            # 6. 更宽松的模糊匹配 - 检查部分字符串
            loose_matched = set()
            if len(adt_found_in_ppi) < len(protein_to_idx) * 0.1:  # 如果匹配率低于10%
                print("警告: 匹配率太低，尝试更宽松的匹配...")
                
                for protein in adata.var_names:
                    if protein not in adt_found_in_ppi:
                        # 如果蛋白质名称长度大于3，尝试匹配部分字符串
                        if len(protein) > 3:
                            for string_id in ppi_network.keys():
                                # 检查STRING ID是否包含蛋白质名称的前3个字符
                                if protein[:3].lower() in string_id.lower():
                                    string_to_adt[string_id] = protein
                                    adt_found_in_ppi.add(protein)
                                    loose_matched.add(protein)
                                    break
            
            print(f"宽松匹配到的蛋白质数量: {len(loose_matched)}")
            if len(loose_matched) > 0:
                print(f"宽松匹配的蛋白质示例: {list(loose_matched)[:5]}")
        
        print(f"总共匹配到的蛋白质: {len(adt_found_in_ppi)}")
        print(f"匹配率: {len(adt_found_in_ppi) / len(protein_to_idx):.2%}")
        
        # 输出所有匹配到的蛋白质名称和对应的STRING ID
        print("\n匹配到的蛋白质与STRING ID对应关系:")
        # 反向映射，找出每个ADT蛋白质对应的STRING ID
        adt_to_string_ids = {}
        for string_id, adt_protein in string_to_adt.items():
            if adt_protein not in adt_to_string_ids:
                adt_to_string_ids[adt_protein] = []
            adt_to_string_ids[adt_protein].append(string_id)
        
        # 按ADT蛋白质名称排序输出
        sorted_adt_proteins = sorted(adt_to_string_ids.keys())
        for i, adt_protein in enumerate(sorted_adt_proteins):
            string_ids = adt_to_string_ids[adt_protein]
            # 限制输出数量，避免输出过多
            if i < 20:  # 只输出前20个
                print(f"  ADT蛋白质: {adt_protein} -> STRING ID: {string_ids[:3]}{'...' if len(string_ids) > 3 else ''}")
        
        if len(sorted_adt_proteins) > 20:
            print(f"  ... 还有 {len(sorted_adt_proteins) - 20} 个匹配关系未显示")
        
        # 如果没有匹配到任何蛋白质，无法进行填补
        if not string_to_adt:
            print("错误: 没有匹配到任何蛋白质，无法进行填补")
            return adata
        
        # 反向映射: ADT蛋白质 -> STRING ID列表
        adt_to_string = {}
        for string_id, adt_protein in string_to_adt.items():
            if adt_protein not in adt_to_string:
                adt_to_string[adt_protein] = []
            adt_to_string[adt_protein].append(string_id)
        
        # 找出所有可预测的蛋白质（在PPI网络中的蛋白质）
        predictable_proteins = set()
        # 使用排序确保遍历顺序的一致性
        for adt_protein in sorted(adt_to_string.keys()):
            string_ids = adt_to_string[adt_protein]
            for string_id in sorted(string_ids):
                if string_id in ppi_network:
                    for p2 in sorted(ppi_network[string_id].keys()):
                        if p2 in string_to_adt:
                            predictable_proteins.add(string_to_adt[p2])
        
        print(f"在PPI网络中找到{len(predictable_proteins)}个可预测的蛋白质")
        # 排序后显示，确保输出一致性
        sorted_predictable = sorted(list(predictable_proteins))
        print(f"可预测蛋白质示例: {sorted_predictable[:10]}")
        
        # 计算每个蛋白质的表达频率
        protein_expression_freq = np.sum(X_orig > 0, axis=0)
        
        # 找出需要填补的蛋白质（在可预测蛋白质中且表达频率较低的蛋白质）
        proteins_to_impute = []
        # 使用排序确保处理顺序的一致性
        for protein in sorted(predictable_proteins):
            protein_idx = protein_to_idx[protein]
            if protein_expression_freq[protein_idx] < np.mean(protein_expression_freq):
                proteins_to_impute.append(protein_idx)
        
        print(f"找到{len(proteins_to_impute)}个需要填补的蛋白质")
        print(f"需要填补的蛋白质示例: {[adata.var_names[idx] for idx in proteins_to_impute[:10]]}")
        
        # 对每个需要填补的蛋白质进行处理
        filled_proteins = 0
        filled_values = 0
        filled_protein_names = []
        
        # 如果使用RNA数据，需要建立基因名到索引的映射
        gene_to_idx = {}
        if use_rna_data and rna_adata is not None:
            gene_to_idx = {gene: i for i, gene in enumerate(rna_adata.var_names)}
            print(f"RNA数据中的基因数量: {len(gene_to_idx)}")
        
        # 计算每个蛋白质的零值数量，用于控制填补率
        protein_zeros = np.sum(X_orig == 0, axis=0)
        
        # 创建跟踪填补率的字典
        protein_impute_rates = {}
        
        # 对每个蛋白质的总样本数
        total_samples = X_orig.shape[0]
        
        for protein_idx in tqdm(proteins_to_impute, desc="填补蛋白质表达"):
            protein_name = adata.var_names[protein_idx]
            
            # 找出与该蛋白质相互作用的蛋白质
            interacting_proteins = []
            interaction_weights = []
            
            # 获取该蛋白质对应的STRING ID
            string_ids = adt_to_string.get(protein_name, [])
            
            # 添加调试信息（排序后输出确保一致性）
            print(f"处理蛋白质: {protein_name}, STRING IDs: {sorted(string_ids)}")
            
            # 排序STRING IDs确保处理顺序一致性
            for string_id in sorted(string_ids):
                if string_id in ppi_network:
                    print(f"  {string_id} 在PPI网络中，有 {len(ppi_network[string_id])} 个互作")
                    
                    # 1. 首先尝试在ADT数据中找互作蛋白质
                    adt_interactors = 0
                    # 排序PPI网络中的相互作用，确保处理顺序一致性
                    for p2 in sorted(ppi_network[string_id].keys()):
                        conf = ppi_network[string_id][p2]
                        if conf >= confidence_threshold and p2 in string_to_adt:
                            interacting_proteins.append(string_to_adt[p2])
                            interaction_weights.append(conf)
                            adt_interactors += 1
                    
                    print(f"  找到 {adt_interactors} 个ADT数据中的互作蛋白质 (置信度阈值 >= {confidence_threshold})")
                    
                    # 2. 如果在ADT中找不到足够的互作蛋白质，尝试使用RNA数据
                    if adt_interactors == 0 and use_rna_data and rna_adata is not None:
                        rna_interactors = 0
                        # 排序PPI网络中的相互作用，确保处理顺序一致性
                        for p2 in sorted(ppi_network[string_id].keys()):
                            conf = ppi_network[string_id][p2]
                            if conf >= confidence_threshold:
                                # 从STRING ID提取基因符号
                                gene_symbol = None
                                if '.' in p2:  # 如果是形如"ENSP00000350844"或"9606.ENSP00000350844"的ID
                                    if hasattr(self, 'protein_id_mapper') and self.protein_id_mapper:
                                        gene_symbol = self.protein_id_mapper.map_string_id_to_protein(p2)
                                
                                # 检查基因符号是否在RNA数据中
                                if gene_symbol and gene_symbol in gene_to_idx:
                                    # 使用RNA数据中的表达值
                                    gene_idx = gene_to_idx[gene_symbol]
                                    rna_expr = rna_adata.X[:, gene_idx].toarray().flatten() if scipy.sparse.issparse(rna_adata.X) else rna_adata.X[:, gene_idx]
                                    
                                    # 只有当RNA表达不全为零时才考虑
                                    if np.any(rna_expr > 0):
                                        # 创建一个虚拟的"ADT蛋白质"标识符
                                        virtual_protein = f"RNA_{gene_symbol}"
                                        # 添加到互作列表
                                        interacting_proteins.append(virtual_protein)
                                        interaction_weights.append(conf)
                                        # 添加表达值到X_orig中的一个临时位置
                                        if virtual_protein not in protein_to_idx:
                                            protein_to_idx[virtual_protein] = -1  # 标记为临时
                                            X_orig = np.hstack((X_orig, rna_expr.reshape(-1, 1)))
                                            protein_to_idx[virtual_protein] = X_orig.shape[1] - 1
                                        rna_interactors += 1
                        
                        print(f"  找到 {rna_interactors} 个RNA数据中的互作基因")
                    
                    # 3. 如果仍然找不到互作伙伴，尝试二级互作
                    if len(interacting_proteins) == 0:
                        print("  尝试查找二级互作蛋白质...")
                        secondary_interactors = 0
                        # 对于每个互作蛋白质，排序确保一致性
                        for p2 in sorted(ppi_network[string_id].keys()):
                            conf1 = ppi_network[string_id][p2]
                            if conf1 >= confidence_threshold:
                                # 查找它的互作蛋白质
                                if p2 in ppi_network:
                                    # 排序二级互作，确保一致性
                                    for p3 in sorted(ppi_network[p2].keys()):
                                        conf2 = ppi_network[p2][p3]
                                        if conf2 >= confidence_threshold and p3 in string_to_adt:
                                            # 使用组合置信度
                                            combined_conf = conf1 * conf2
                                            if combined_conf >= confidence_threshold:
                                                interacting_proteins.append(string_to_adt[p3])
                                                interaction_weights.append(combined_conf)
                                                secondary_interactors += 1
                        
                        print(f"  找到 {secondary_interactors} 个二级互作蛋白质")
            
            if not interacting_proteins:
                print(f"  未找到互作蛋白质，跳过")
                continue
            
            print(f"  找到 {len(interacting_proteins)} 个互作蛋白质，置信度阈值: {confidence_threshold}")
            
            # 计算加权平均表达值和置信度
            weighted_expr = np.zeros(X_orig.shape[0])
            confidence_scores = np.zeros(X_orig.shape[0])  # 用于存储每个位置的置信度
            total_weight = 0
            
            for p2, weight in zip(interacting_proteins, interaction_weights):
                p2_idx = protein_to_idx[p2]
                weighted_expr += weight * X_orig[:, p2_idx]
                confidence_scores += weight  # 累加权重作为置信度
                total_weight += weight
            
            if total_weight > 0:
                predicted_expr = weighted_expr / total_weight
                confidence_scores = confidence_scores / total_weight  # 归一化置信度
                
                # 只填补原始数据中为零且置信度高于阈值的位置
                original_zero_mask = X_orig[:, protein_idx] == 0
                high_confidence_mask = confidence_scores >= min_confidence_for_imputation
                
                # 找出所有可能填补的位置（零值且预测值大于0）
                potential_fill_mask = original_zero_mask & high_confidence_mask & (predicted_expr > 0)
                potential_fill_positions = np.where(potential_fill_mask)[0]
                potential_fill_values = predicted_expr[potential_fill_positions]
                
                # 只选择预测值最高的前N%位置
                if len(potential_fill_positions) > 0:
                    # 计算要选择的位置数量
                    select_count = max(1, int(len(potential_fill_positions) * top_percent_to_impute / 100))
                    
                    # 按预测值大小排序，选择最高的
                    sorted_indices = np.argsort(potential_fill_values)[::-1]  # 降序排列
                    top_indices = sorted_indices[:select_count]
                    top_positions = potential_fill_positions[top_indices]
                    
                    # 创建最终的填补掩码
                    fillable_mask = np.zeros_like(original_zero_mask, dtype=bool)
                    fillable_mask[top_positions] = True
                else:
                    fillable_mask = potential_fill_mask
                
                zero_count = np.sum(original_zero_mask)
                high_conf_count = np.sum(original_zero_mask & high_confidence_mask)
                print(f"  原始零值数量: {zero_count}")
                print(f"  高置信度零值数量: {high_conf_count}")
                
                # 检查预测值
                pred_nonzero = np.sum(predicted_expr > 0)
                print(f"  预测非零值数量: {pred_nonzero}")
                
                # 计算可填补的值数量
                potential_changes = np.sum(fillable_mask)
                print(f"  可填补的值数量: {potential_changes}")
                print(f"  选择了预测值最高的前{top_percent_to_impute}%位置，共{potential_changes}个")
                
                # 计算填补率并应用上限
                if zero_count > 0:
                    # 计算真实填补率 = 填补值数量/总样本数量
                    impute_rate = potential_changes / total_samples
                    print(f"  潜在填补率(相对于总样本数): {impute_rate:.2%}")
                    
                    # 如果填补率超过上限，选择预测值较高的位置进行填补
                    if impute_rate > max_impute_rate and potential_changes > 0:
                        print(f"  填补率超过上限 {max_impute_rate*100}%，将选择预测值较高的位置进行填补")
                        
                        # 找出所有可填补的位置
                        fillable_positions = np.where(fillable_mask)[0]
                        
                        # 获取这些位置的预测值
                        fillable_values = predicted_expr[fillable_positions]
                        
                        # 计算允许填补的最大数量
                        max_fill_count = int(total_samples * max_impute_rate)
                        
                        # 根据预测值大小排序，选择预测值最高的位置
                        sorted_indices = np.argsort(fillable_values)[::-1]  # 降序排列
                        selected_indices = sorted_indices[:min(max_fill_count, len(fillable_positions))]
                        selected_positions = fillable_positions[selected_indices]
                        
                        # 创建新的掩码，只在选定的位置填补
                        new_mask = np.zeros_like(original_zero_mask, dtype=bool)
                        new_mask[selected_positions] = True
                        
                        # 应用新掩码
                        changes = np.sum(new_mask)
                        X_orig[new_mask, protein_idx] = predicted_expr[new_mask]
                        
                        # 记录实际填补率
                        actual_impute_rate = changes / total_samples
                        protein_impute_rates[protein_name] = actual_impute_rate * 100
                        
                        print(f"  实际填补 {changes} 个值，填补率(相对于总样本数): {actual_impute_rate:.2%}")
                        print(f"  填补值的平均大小: {np.mean(predicted_expr[new_mask]):.4f}")
                    else:
                        # 填补率在允许范围内，直接填补
                        changes = potential_changes
                        #X_orig[original_zero_mask & (predicted_expr > 0), protein_idx] = predicted_expr[original_zero_mask & (predicted_expr > 0)]
                        X_orig[fillable_mask, protein_idx] = predicted_expr[fillable_mask]
                        
                        # 记录实际填补率
                        actual_impute_rate = changes / total_samples
                        protein_impute_rates[protein_name] = actual_impute_rate * 100
                        
                        print(f"  填补 {changes} 个值，填补率(相对于总样本数): {actual_impute_rate:.2%}")
                    
                    if changes > 0:
                        filled_proteins += 1
                        filled_values += changes
                        filled_protein_names.append(protein_name)
                else:
                    print(f"  无零值需要填补")
                    protein_impute_rates[protein_name] = 0.0
            else:
                print(f"  无法计算预测值，跳过")
                protein_impute_rates[protein_name] = 0.0
        
        # 保存原始数据备份用于计算插补信息
        X_original_backup = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X.copy()
        
        # 创建插补掩码（标记哪些位置被插补了）
        imputation_mask = np.zeros_like(X_orig, dtype=bool)
        for protein_name in filled_protein_names:
            if protein_name in adata.var_names:
                protein_idx = protein_to_idx[protein_name]
                # 找出原始为0但现在非0的位置
                original_zero = X_original_backup[:, protein_idx] == 0
                current_nonzero = X_orig[:, protein_idx] > 0
                imputation_mask[:, protein_idx] = original_zero & current_nonzero
        
        # 创建仅包含插补值的数据矩阵
        X_imputed_only = np.zeros_like(X_orig)
        X_imputed_only[imputation_mask] = X_orig[imputation_mask]
        
        # 创建AnnData对象
        adata_imputed_full = adata.copy()
        adata_imputed_full.X = scipy.sparse.csr_matrix(X_orig)
        
        adata_imputed_only = adata.copy()
        adata_imputed_only.X = scipy.sparse.csr_matrix(X_imputed_only)
        
        # 将填补率信息添加到adata.var中
        impute_rates = np.zeros(adata.n_vars)
        is_imputed = np.zeros(adata.n_vars, dtype=bool)
        
        for protein, rate in protein_impute_rates.items():
            if protein in adata.var_names:
                idx = np.where(adata.var_names == protein)[0][0]
                impute_rates[idx] = rate
                is_imputed[idx] = protein in filled_protein_names
        
        adata_imputed_full.var['impute_rate'] = impute_rates
        adata_imputed_full.var['is_imputed'] = is_imputed
        adata_imputed_only.var['impute_rate'] = impute_rates
        adata_imputed_only.var['is_imputed'] = is_imputed
        
        # 创建插补信息字典
        imputation_info = {
            'method': 'PPI_network',
            'total_proteins_processed': len(proteins_to_impute),
            'successful_proteins': filled_proteins,
            'total_filled_values': filled_values,
            'confidence_threshold': confidence_threshold,
            'max_impute_rate': max_impute_rate,
            'min_confidence_for_imputation': min_confidence_for_imputation,
            'top_percent_to_impute': top_percent_to_impute,
            'imputation_rate': filled_values / X_orig.size,
            'proteins_to_impute': [adata_imputed_full.var_names[idx] for idx in proteins_to_impute],
            'imputed_protein_names': filled_protein_names,
            'protein_impute_rates': protein_impute_rates,
            'imputation_mask': imputation_mask,
            'original_data_shape': X_original_backup.shape,
            'imputed_data_shape': X_imputed_only.shape,
            'matched_proteins_count': len(adt_found_in_ppi),
            'total_proteins_in_data': len(protein_to_idx)
        }
        
        # 计算填补效果
        print(f"\n填补效果统计:")
        print(f"- 成功填补了{filled_proteins}个蛋白质")
        print(f"- 总共填补了{filled_values}个值")
        print(f"- 填补后非零元素比例: {np.count_nonzero(X_orig) / X_orig.size:.2%}")
        print(f"- 总体插补率: {imputation_info['imputation_rate']:.4f}")
        
        # 输出成功填补的蛋白质名称和填补率
        print(f"\n成功填补的蛋白质详细信息:")
        print(f"{'序号':<4} {'蛋白质名称':<15} {'插补率(%)':<10} {'插补细胞数':<10} {'映射来源':<12}")
        print("-" * 60)
        
        # 按填补率排序
        sorted_proteins_by_rate = sorted(protein_impute_rates.items(), key=lambda x: x[1], reverse=True)
        
        # 输出前15个填补率最高的蛋白质
        for i, (protein, rate) in enumerate(sorted_proteins_by_rate[:15], 1):
            source = "unknown"
            
            # 获取映射来源
            if 'mapping_source' in adata.var.columns:
                idx = np.where(adata.var_names == protein)[0]
                if len(idx) > 0:
                    source = adata.var['mapping_source'].iloc[idx[0]]
            
            # 计算插补细胞数
            if protein in protein_to_idx:
                protein_idx = protein_to_idx[protein]
                imputed_cells = np.sum(imputation_mask[:, protein_idx])
            else:
                imputed_cells = 0
            
            print(f"{i:<4} {protein:<15} {rate:<10.2f} {imputed_cells:<10} {source:<12}")
        
        if len(sorted_proteins_by_rate) > 15:
            print(f"... 还有 {len(sorted_proteins_by_rate) - 15} 个蛋白质未显示")
        
        return adata_imputed_full, adata_imputed_only, imputation_info


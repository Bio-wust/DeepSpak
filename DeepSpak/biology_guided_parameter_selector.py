import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BiologyGuidedParameterSelector:
    """
    基于生物学原理的插补参数选择器
    利用数据的内在生物学特征来自动确定最优插补参数
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def log(self, message):
        if self.verbose:
            print(f"[BiologySelector] {message}")
            
    def calculate_grn_confidence_from_co_expression(self, adata_rna):
        """
        基于基因共表达网络计算GRN置信度阈值
        
        原理：
        - 真正的调控关系应该在共表达网络中表现出强相关性
        - 通过分析基因间相关性分布，确定高置信度阈值
        """
        self.log("计算基于共表达的GRN置信度阈值...")
        
        # 获取表达矩阵
        X = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X
        
        # 过滤低表达基因，减少噪音
        gene_mean_expr = np.mean(X, axis=0)
        high_expr_mask = gene_mean_expr > np.percentile(gene_mean_expr, 25)
        X_filtered = X[:, high_expr_mask]
        
        self.log(f"使用{np.sum(high_expr_mask)}个高表达基因计算相关性")
        
        # 计算基因间相关性矩阵（使用Spearman相关，对outlier更robust）
        n_genes = X_filtered.shape[1]
        if n_genes > 1000:  # 如果基因太多，随机采样以提高效率
            sample_indices = np.random.choice(n_genes, 1000, replace=False)
            X_sample = X_filtered[:, sample_indices]
        else:
            X_sample = X_filtered
            
        # 计算相关性
        correlations = []
        n_genes_sample = X_sample.shape[1]
        
        for i in tqdm(range(n_genes_sample), desc="计算基因相关性", disable=not self.verbose):
            for j in range(i+1, n_genes_sample):
                corr, p_val = spearmanr(X_sample[:, i], X_sample[:, j])
                if not np.isnan(corr) and p_val < 0.05:  # 只考虑显著相关的基因对
                    correlations.append(abs(corr))
        
        correlations = np.array(correlations)
        
        # 基于相关性分布确定置信度阈值
        # 取75%分位数作为高置信度阈值，这样可以筛选出真正有生物学意义的调控关系
        confidence_threshold = np.percentile(correlations, 75)
        
        # 确保阈值在合理范围内
        confidence_threshold = max(0.3, min(0.8, confidence_threshold))
        
        self.log(f"基于共表达分析的GRN置信度阈值: {confidence_threshold:.3f}")
        return confidence_threshold
    
    def estimate_min_cells_from_gene_distribution(self, adata_rna):
        """
        基于基因表达分布估计最小表达细胞数
        
        改进策略：
        1. 使用统计显著性而非经验阈值
        2. 基于数据分布特征自适应调整
        3. 减少硬编码的经验参数
        """
        self.log("基于数据分布估计最小表达细胞数...")
        
        # 获取表达矩阵
        X = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X
        
        # 计算每个基因的表达细胞数
        expressing_cells_per_gene = np.sum(X > 0, axis=0)
        total_cells = X.shape[0]
        
        # 分析表达细胞数的分布特征
        median_expressing = np.median(expressing_cells_per_gene)
        q25_expressing = np.percentile(expressing_cells_per_gene, 25)
        q75_expressing = np.percentile(expressing_cells_per_gene, 75)
        
        # 计算分布的统计特征
        mean_expressing = np.mean(expressing_cells_per_gene)
        std_expressing = np.std(expressing_cells_per_gene)
        
        # 方法1: 基于分布的分位数自适应确定
        # 使用Q25作为基础，但根据分布形状调整
        distribution_skew = (mean_expressing - median_expressing) / (std_expressing + 1e-8)
        
        # 根据分布偏度调整系数
        if abs(distribution_skew) < 0.5:  # 接近正态分布
            base_coefficient = 0.3
        elif distribution_skew > 0:  # 右偏分布（长尾）
            base_coefficient = 0.2  # 更保守
        else:  # 左偏分布
            base_coefficient = 0.4  # 更宽松
        
        min_cells_method1 = max(1, int(q25_expressing * base_coefficient))
        
        # 方法2: 基于统计显著性
        # 计算表达细胞数的显著性阈值（基于正态分布假设）
        significant_threshold = max(1, int(mean_expressing - 1.5 * std_expressing))
        min_cells_method2 = max(1, significant_threshold)
        
        # 方法3: 基于数据密度自适应
        # 计算基因表达的整体密度
        overall_density = np.sum(X > 0) / (X.shape[0] * X.shape[1])
        
        # 根据密度自适应调整
        if overall_density < 0.05:  # 极稀疏
            density_coefficient = 0.1
        elif overall_density < 0.15:  # 稀疏
            density_coefficient = 0.2
        elif overall_density < 0.3:  # 中等
            density_coefficient = 0.3
        else:  # 密集
            density_coefficient = 0.4
            
        min_cells_method3 = max(1, int(median_expressing * density_coefficient))
        
        # 综合三种方法，取中位数避免极端值
        candidates = [min_cells_method1, min_cells_method2, min_cells_method3]
        min_cells = int(np.median(candidates))
        
        # 最终验证：确保在合理范围内
        # 下限：至少1个细胞（生物学合理性）
        # 上限：不超过总细胞数的20%（避免过度过滤）
        min_cells = max(1, min(min_cells, int(total_cells * 0.2)))
        
        # 记录详细信息
        self.log(f"数据统计:")
        self.log(f"  - 总细胞数: {total_cells}")
        self.log(f"  - 基因表达中位数: {median_expressing:.1f}")
        self.log(f"  - 基因表达Q25: {q25_expressing:.1f}")
        self.log(f"  - 基因表达Q75: {q75_expressing:.1f}")
        self.log(f"  - 分布偏度: {distribution_skew:.3f}")
        self.log(f"  - 整体密度: {overall_density:.3f}")
        
        self.log(f"三种方法结果:")
        self.log(f"  - 方法1(分位数): {min_cells_method1}")
        self.log(f"  - 方法2(显著性): {min_cells_method2}")
        self.log(f"  - 方法3(密度): {min_cells_method3}")
        self.log(f"最终选择: {min_cells}")
        
        return min_cells
    
    def calculate_ppi_confidence_from_protein_abundance(self, adata_adt):
        """
        基于蛋白质丰度计算PPI网络置信度阈值
        
        原理：
        - 高丰度蛋白质的相互作用更可能被检测到
        - 基于蛋白质表达水平的分布来确定PPI置信度阈值
        """
        self.log("计算基于蛋白质丰度的PPI置信度阈值...")
        
        # 获取表达矩阵
        X = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X
        
        # 计算蛋白质平均表达水平
        protein_mean_expr = np.mean(X, axis=0)
        protein_var = np.var(X, axis=0)
        
        # 计算变异系数 (CV = std/mean)
        cv = np.sqrt(protein_var) / (protein_mean_expr + 1e-6)
        
        # 高表达且低变异的蛋白质更可能有稳定的相互作用
        # 基于表达水平和稳定性确定置信度阈值
        high_expr_mask = protein_mean_expr > np.percentile(protein_mean_expr, 50)
        stable_mask = cv < np.percentile(cv, 50)
        reliable_proteins = high_expr_mask & stable_mask
        
        reliable_ratio = np.sum(reliable_proteins) / len(reliable_proteins)
        
        # 基于可靠蛋白质比例确定PPI置信度阈值
        if reliable_ratio > 0.5:  # 大部分蛋白质都可靠
            ppi_confidence = 0.8
        elif reliable_ratio > 0.3:  # 中等可靠性
            ppi_confidence = 0.7
        else:  # 低可靠性，需要更严格的阈值
            ppi_confidence = 0.85
            
        self.log(f"可靠蛋白质比例: {reliable_ratio:.3f}")
        self.log(f"PPI网络置信度阈值: {ppi_confidence}")
        return ppi_confidence
    
    def estimate_impute_rate_from_dropout_pattern(self, adata_adt):
        """
        基于dropout模式估计最大插补比例
        
        原理：
        - 分析数据的dropout模式，避免过度插补
        - 基于零值分布确定合理的插补上限
        """
        self.log("估计基于dropout模式的最大插补比例...")
        
        # 获取表达矩阵
        X = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X
        
        # 计算零值比例
        zero_ratio = np.sum(X == 0) / X.size
        
        # 分析每个蛋白质的dropout程度
        protein_dropout = np.sum(X == 0, axis=0) / X.shape[0]
        median_dropout = np.median(protein_dropout)
        
        # 基于dropout程度确定最大插补比例
        # dropout越严重，允许的插补比例越大，但有上限
        if median_dropout > 0.7:  # 高dropout
            max_impute_rate = min(0.1, median_dropout * 0.15)
        elif median_dropout > 0.5:  # 中等dropout
            max_impute_rate = min(0.08, median_dropout * 0.12)
        else:  # 低dropout
            max_impute_rate = min(0.05, median_dropout * 0.1)
            
        # 确保在合理范围内
        max_impute_rate = max(0.01, min(0.2, max_impute_rate))
        
        self.log(f"数据零值比例: {zero_ratio:.3f}")
        self.log(f"蛋白质中位dropout: {median_dropout:.3f}")
        self.log(f"最大插补比例: {max_impute_rate:.3f}")
        return max_impute_rate
    
    def calculate_min_confidence_from_noise_level(self, adata_adt):
        """
        基于噪音水平计算最小插补置信度
        
        原理：
        - 估计数据的噪音水平
        - 设置足够高的置信度阈值以避免插补噪音
        """
        self.log("计算基于噪音水平的最小插补置信度...")
        
        # 获取表达矩阵
        X = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X
        
        # 估计技术噪音水平
        # 方法1: 基于低表达蛋白质的变异性
        protein_mean = np.mean(X, axis=0)
        protein_std = np.std(X, axis=0)
        
        # 低表达蛋白质的变异主要来自技术噪音
        low_expr_mask = protein_mean < np.percentile(protein_mean, 25)
        if np.sum(low_expr_mask) > 0:
            noise_level = np.median(protein_std[low_expr_mask] / (protein_mean[low_expr_mask] + 1e-6))
        else:
            noise_level = np.median(protein_std / (protein_mean + 1e-6))
        
        # 基于噪音水平确定最小置信度
        if noise_level > 1.0:  # 高噪音
            min_confidence = 0.85
        elif noise_level > 0.5:  # 中等噪音
            min_confidence = 0.75
        else:  # 低噪音
            min_confidence = 0.65
            
        # 确保在合理范围内
        min_confidence = max(0.5, min(0.9, min_confidence))
        
        self.log(f"估计的噪音水平: {noise_level:.3f}")
        self.log(f"最小插补置信度: {min_confidence:.3f}")
        return min_confidence
    
    def estimate_top_percent_from_expression_variance(self, adata_adt):
        """
        基于表达变异性估计插补前N%位置
        
        原理：
        - 高变异性表明表达水平的不确定性更大
        - 只对最有把握的位置进行插补
        """
        self.log("估计基于表达变异性的插补百分比...")
        
        # 获取表达矩阵
        X = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X
        
        # 计算表达变异性
        protein_mean = np.mean(X, axis=0)
        protein_var = np.var(X, axis=0)
        
        # 计算变异系数
        cv = np.sqrt(protein_var) / (protein_mean + 1e-6)
        median_cv = np.median(cv)
        
        # 基于变异性确定插补百分比
        # 变异性越大，只插补更少的top位置
        if median_cv > 2.0:  # 高变异性
            top_percent = 5
        elif median_cv > 1.0:  # 中等变异性
            top_percent = 10
        else:  # 低变异性
            top_percent = 15
            
        # 额外考虑数据质量
        # 如果大部分蛋白质表达都很低，减少插补比例
        low_expr_ratio = np.sum(protein_mean < np.percentile(protein_mean, 50)) / len(protein_mean)
        if low_expr_ratio > 0.7:
            top_percent = max(3, top_percent - 5)
            
        # 确保在合理范围内
        top_percent = max(1, min(30, top_percent))
        
        self.log(f"表达变异系数中位数: {median_cv:.3f}")
        self.log(f"低表达蛋白质比例: {low_expr_ratio:.3f}")
        self.log(f"插补前百分比: {top_percent}%")
        return top_percent
    
    def calculate_atac_confidence_from_peak_tf_correlation(self, adata_rna, adata_atac):
        """
        基于Peak-TF相关性计算ATAC插补置信度阈值
        
        原理：
        - 分析转录因子基因表达与对应ATAC峰的相关性
        - 高相关性表明Peak-TF映射关系的可靠性
        """
        self.log("计算基于Peak-TF相关性的ATAC置信度阈值...")
        
        # 获取表达矩阵
        X_rna = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X
        X_atac = adata_atac.X.toarray() if scipy.sparse.issparse(adata_atac.X) else adata_atac.X
        
        # 如果基因名和峰名有重叠，计算它们的相关性
        rna_genes = set(adata_rna.var_names)
        atac_peaks = set(adata_atac.var_names)
        
        # 寻找转录因子基因（通常在基因名中包含特定模式）
        tf_genes = []
        for gene in rna_genes:
            # 常见的转录因子基因模式
            if any(pattern in gene.upper() for pattern in ['TF', 'FOX', 'SOX', 'HOX', 'PAX', 'MYC', 'MAX']):
                tf_genes.append(gene)
        
        if len(tf_genes) < 10:
            # 如果找不到足够的TF，使用高变异基因作为代理
            if 'highly_variable' not in adata_rna.var.columns:
                sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=2000)
            tf_genes = adata_rna.var_names[adata_rna.var['highly_variable']][:200].tolist()
        
        # 计算TF基因与ATAC峰的相关性
        correlations = []
        for tf_gene in tf_genes[:100]:  # 限制计算量
            if tf_gene in rna_genes:
                rna_idx = list(adata_rna.var_names).index(tf_gene)
                
                # 随机采样一些峰来计算相关性
                sample_peaks = np.random.choice(len(adata_atac.var_names), 
                                              min(500, len(adata_atac.var_names)), 
                                              replace=False)
                
                for peak_idx in sample_peaks:
                    corr, p_val = spearmanr(X_rna[:, rna_idx], X_atac[:, peak_idx])
                    if not np.isnan(corr) and p_val < 0.05:
                        correlations.append(abs(corr))
        
        if len(correlations) == 0:
            self.log("警告：未找到显著的Peak-TF相关性，使用默认阈值")
            return 0.4
            
        correlations = np.array(correlations)
        
        # 基于相关性分布确定置信度阈值
        # 使用70%分位数，比GRN稍微宽松一些，因为Peak-TF关系更复杂
        confidence_threshold = np.percentile(correlations, 70)
        
        # 确保阈值在合理范围内
        confidence_threshold = max(0.2, min(0.7, confidence_threshold))
        
        self.log(f"计算了{len(correlations)}个Peak-TF相关性")
        self.log(f"基于Peak-TF相关性的ATAC置信度阈值: {confidence_threshold:.3f}")
        return confidence_threshold
    
    def estimate_atac_min_cells_from_accessibility_pattern(self, adata_atac):
        """
        基于染色质可及性模式估计ATAC最小细胞数
        
        原理：
        - ATAC数据通常比RNA数据更稀疏
        - 基于峰的开放性分布来确定最小细胞数阈值
        """
        self.log("估计基于可及性模式的ATAC最小细胞数...")
        
        # 获取表达矩阵
        X = adata_atac.X.toarray() if scipy.sparse.issparse(adata_atac.X) else adata_atac.X
        
        # 计算每个峰的开放细胞数
        accessible_cells_per_peak = np.sum(X > 0, axis=0)
        
        # 分析可及性分布
        median_accessible = np.median(accessible_cells_per_peak)
        q25_accessible = np.percentile(accessible_cells_per_peak, 25)
        q10_accessible = np.percentile(accessible_cells_per_peak, 10)
        
        # ATAC数据通常比RNA更稀疏，使用更小的阈值
        total_cells = X.shape[0]
        accessibility_ratio = median_accessible / total_cells
        
        if accessibility_ratio < 0.05:  # 极高稀疏性
            min_cells = max(2, int(q10_accessible * 0.8))
        elif accessibility_ratio < 0.15:  # 高稀疏性
            min_cells = max(3, int(q25_accessible * 0.6))
        elif accessibility_ratio < 0.3:  # 中等稀疏性
            min_cells = max(5, int(median_accessible * 0.4))
        else:  # 低稀疏性
            min_cells = max(8, int(median_accessible * 0.5))
            
        # 确保最小细胞数在合理范围内
        min_cells = max(2, min(30, min_cells))
        
        self.log(f"可及性比例: {accessibility_ratio:.3f}")
        self.log(f"估计的ATAC最小开放细胞数: {min_cells}")
        return min_cells
    
    def biology_guided_parameter_selection(self, adata_rna, adata_adt=None, adata_atac=None):
        """
        基于生物学特征自动选择插补参数的主函数
        
        参数:
            adata_rna: RNA数据的AnnData对象
            adata_adt: ADT/蛋白质数据的AnnData对象（可选）
            adata_atac: ATAC数据的AnnData对象（可选）
            
        返回:
            结果字典，包含可用的参数集合
        """
        self.log("开始基于生物学特征的参数选择...")
        
        results = {}
        
        # RNA参数计算
        self.log("\n=== RNA参数计算 ===")
        results['rna_params'] = {
            'confidence_threshold': self.calculate_grn_confidence_from_co_expression(adata_rna),
            'min_cells': self.estimate_min_cells_from_gene_distribution(adata_rna)
        }
        
        # ADT参数计算（如果提供了ADT数据）
        if adata_adt is not None:
            self.log("\n=== ADT参数计算 ===")
            results['adt_params'] = {
                'confidence_threshold': self.calculate_ppi_confidence_from_protein_abundance(adata_adt),
                'max_impute_rate': self.estimate_impute_rate_from_dropout_pattern(adata_adt),
                'min_confidence_for_imputation': self.calculate_min_confidence_from_noise_level(adata_adt),
                'top_percent_to_impute': self.estimate_top_percent_from_expression_variance(adata_adt)
            }
        
        # ATAC参数计算（如果提供了ATAC数据）
        if adata_atac is not None:
            self.log("\n=== ATAC参数计算 ===")
            results['atac_params'] = {
                'confidence_threshold': self.calculate_atac_confidence_from_peak_tf_correlation(adata_rna, adata_atac),
                'min_cells': self.estimate_atac_min_cells_from_accessibility_pattern(adata_atac)
            }
        
        # 输出最终参数
        self.log("\n=== 最终参数选择结果 ===")
        for data_type, params in results.items():
            self.log(f"{data_type.upper()}:")
            for key, value in params.items():
                self.log(f"  {key}: {value}")
        
        return results
    
    def get_parameter_explanation(self, adata_rna, adata_adt=None, adata_atac=None):
        """
        提供参数选择的详细解释和数据质量评估
        """
        X_rna = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X
        
        report = {
            'data_quality': {
                'rna_sparsity': np.sum(X_rna == 0) / X_rna.size,
                'rna_genes': X_rna.shape[1],
                'cells': X_rna.shape[0]
            },
            'parameter_rationale': {
                'rna_confidence': "基于基因共表达网络相关性分布的75%分位数",
                'rna_min_cells': "基于基因表达分布的稀疏性分析"
            }
        }
        
        if adata_adt is not None:
            X_adt = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X
            report['data_quality'].update({
                'adt_sparsity': np.sum(X_adt == 0) / X_adt.size,
                'adt_proteins': X_adt.shape[1]
            })
            report['parameter_rationale'].update({
                'adt_confidence': "基于蛋白质丰度和稳定性的综合评估",
                'adt_max_rate': "基于dropout模式避免过度插补",
                'adt_min_confidence': "基于噪音水平确保插补质量",
                'adt_top_percent': "基于表达变异性确保插补准确性"
            })
        
        if adata_atac is not None:
            X_atac = adata_atac.X.toarray() if scipy.sparse.issparse(adata_atac.X) else adata_atac.X
            report['data_quality'].update({
                'atac_sparsity': np.sum(X_atac == 0) / X_atac.size,
                'atac_peaks': X_atac.shape[1]
            })
            report['parameter_rationale'].update({
                'atac_confidence': "基于Peak-TF相关性分布的70%分位数",
                'atac_min_cells': "基于染色质可及性模式的稀疏性分析"
            })
        
        return report

# 使用示例函数
def example_usage():
    """
    使用示例 - 支持RNA、ADT和ATAC数据的参数选择
    """
    print("=== 生物学指导的插补参数选择器使用示例 ===\n")
    
    # 创建参数选择器
    selector = BiologyGuidedParameterSelector(verbose=True)
    
    print("支持的数据类型和插补方法：")
    print("1. RNA数据 - 使用GRN网络插补")
    print("2. ADT数据 - 使用PPI网络插补") 
    print("3. ATAC数据 - 使用Peak-TF映射插补\n")
    
    print("使用方法示例：")
    print("\n# 加载数据")
    print("# adata_rna = sc.read_h5ad('your_rna_data.h5ad')")
    print("# adata_adt = sc.read_h5ad('your_adt_data.h5ad')  # 可选")
    print("# adata_atac = sc.read_h5ad('your_atac_data.h5ad')  # 可选")
    
    print("\n# 获取参数（支持任意组合）")
    print("# results = selector.biology_guided_parameter_selection(")
    print("#     adata_rna=adata_rna,")
    print("#     adata_adt=adata_adt,    # 可选")
    print("#     adata_atac=adata_atac   # 可选")
    print("# )")
    
    print("\n# 提取特定数据类型的参数")
    print("# rna_params = results['rna_params']")
    print("# adt_params = results.get('adt_params', {})  # 如果没有ADT数据")
    print("# atac_params = results.get('atac_params', {})  # 如果没有ATAC数据")
    
    print("\n# 直接应用到插补方法")
    print("# RNA插补")
    print("# adata_rna_imputed = data_loader.impute_rna_with_grn(")
    print("#     adata_rna, adata_atac, **rna_params")
    print("# )")
    
    print("\n# ADT插补（如果有ADT数据）")
    print("# if 'adt_params' in results:")
    print("#     adata_adt_imputed = data_loader.impute_protein_with_ppi(")
    print("#         adata_adt, **adt_params")
    print("#     )")
    
    print("\n# ATAC插补（如果有ATAC数据）")
    print("# if 'atac_params' in results:")
    print("#     adata_atac_imputed = data_loader.impute_atac_with_peak_tf(")
    print("#         adata_rna, adata_atac, **atac_params")
    print("#     )")
    
    print("\n# 获取详细报告")
    print("# report = selector.get_parameter_explanation(adata_rna, adata_adt, adata_atac)")
    
    print("\n=== ATAC参数说明 ===")
    print("ATAC插补方法参数：")
    print("- confidence_threshold: Peak-TF映射置信度阈值 (0.2-0.7)")
    print("- min_cells: 峰在最少多少个细胞中开放才考虑插补 (2-30)")
    print("\n这些参数基于以下生物学原理计算：")
    print("1. 分析转录因子基因表达与ATAC峰的相关性")
    print("2. 基于染色质可及性模式确定稀疏性阈值")
    print("3. 考虑ATAC数据通常比RNA数据更稀疏的特点")

if __name__ == "__main__":
    example_usage() 
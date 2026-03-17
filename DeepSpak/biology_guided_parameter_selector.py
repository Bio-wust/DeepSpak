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
    Biology-guided imputation parameter selector
    Automatically determines optimal imputation parameters using intrinsic biological features of the data
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def log(self, message):
        if self.verbose:
            print(f"[BiologySelector] {message}")
            
    def calculate_grn_confidence_from_co_expression(self, adata_rna):
        """
        Calculate GRN confidence threshold based on gene co-expression network

        Principle:
        - True regulatory relationships should show strong correlations in co-expression networks
        - Determine high-confidence threshold by analyzing the distribution of gene correlations
        """
        self.log("Calculating GRN confidence threshold based on co-expression...")

        # Get expression matrix
        X = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X

        # Filter low-expression genes to reduce noise
        gene_mean_expr = np.mean(X, axis=0)
        high_expr_mask = gene_mean_expr > np.percentile(gene_mean_expr, 25)
        X_filtered = X[:, high_expr_mask]

        self.log(f"Using {np.sum(high_expr_mask)} high-expression genes for correlation calculation")

        # Calculate gene correlation matrix (using Spearman correlation, more robust to outliers)
        n_genes = X_filtered.shape[1]
        if n_genes > 1000:  # If too many genes, randomly sample to improve efficiency
            sample_indices = np.random.choice(n_genes, 1000, replace=False)
            X_sample = X_filtered[:, sample_indices]
        else:
            X_sample = X_filtered

        # Calculate correlations
        correlations = []
        n_genes_sample = X_sample.shape[1]

        for i in tqdm(range(n_genes_sample), desc="Calculating gene correlations", disable=not self.verbose):
            for j in range(i+1, n_genes_sample):
                corr, p_val = spearmanr(X_sample[:, i], X_sample[:, j])
                if not np.isnan(corr) and p_val < 0.05:  # Only consider significantly correlated gene pairs
                    correlations.append(abs(corr))

        correlations = np.array(correlations)

        # Determine confidence threshold based on correlation distribution
        # Use 75th percentile as high-confidence threshold to filter for biologically meaningful regulatory relationships
        confidence_threshold = np.percentile(correlations, 75)

        # Ensure threshold is within reasonable range
        confidence_threshold = max(0.3, min(0.8, confidence_threshold))

        self.log(f"GRN confidence threshold based on co-expression analysis: {confidence_threshold:.3f}")
        return confidence_threshold
    
    def estimate_min_cells_from_gene_distribution(self, adata_rna):
        """
        Estimate minimum expressing cells based on gene expression distribution

        Improved strategy:
        1. Use statistical significance instead of empirical thresholds
        2. Adaptive adjustment based on data distribution characteristics
        3. Reduce hardcoded empirical parameters
        """
        self.log("Estimating minimum expressing cells based on data distribution...")

        # Get expression matrix
        X = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X

        # Calculate number of expressing cells per gene
        expressing_cells_per_gene = np.sum(X > 0, axis=0)
        total_cells = X.shape[0]

        # Analyze distribution characteristics of expressing cell counts
        median_expressing = np.median(expressing_cells_per_gene)
        q25_expressing = np.percentile(expressing_cells_per_gene, 25)
        q75_expressing = np.percentile(expressing_cells_per_gene, 75)

        # Calculate statistical features of the distribution
        mean_expressing = np.mean(expressing_cells_per_gene)
        std_expressing = np.std(expressing_cells_per_gene)

        # Method 1: Adaptive determination based on distribution quantiles
        # Use Q25 as base, but adjust according to distribution shape
        distribution_skew = (mean_expressing - median_expressing) / (std_expressing + 1e-8)

        # Adjust coefficient based on distribution skewness
        if abs(distribution_skew) < 0.5:  # Near normal distribution
            base_coefficient = 0.3
        elif distribution_skew > 0:  # Right-skewed distribution (long tail)
            base_coefficient = 0.2  # More conservative
        else:  # Left-skewed distribution
            base_coefficient = 0.4  # More lenient

        min_cells_method1 = max(1, int(q25_expressing * base_coefficient))

        # Method 2: Based on statistical significance
        # Calculate significance threshold for expressing cell counts (assuming normal distribution)
        significant_threshold = max(1, int(mean_expressing - 1.5 * std_expressing))
        min_cells_method2 = max(1, significant_threshold)

        # Method 3: Adaptive based on data density
        # Calculate overall density of gene expression
        overall_density = np.sum(X > 0) / (X.shape[0] * X.shape[1])

        # Adaptive adjustment based on density
        if overall_density < 0.05:  # Extremely sparse
            density_coefficient = 0.1
        elif overall_density < 0.15:  # Sparse
            density_coefficient = 0.2
        elif overall_density < 0.3:  # Moderate
            density_coefficient = 0.3
        else:  # Dense
            density_coefficient = 0.4

        min_cells_method3 = max(1, int(median_expressing * density_coefficient))

        # Combine three methods, take median to avoid extreme values
        candidates = [min_cells_method1, min_cells_method2, min_cells_method3]
        min_cells = int(np.median(candidates))

        # Final validation: ensure within reasonable range
        # Lower bound: at least 1 cell (biological plausibility)
        # Upper bound: no more than 20% of total cells (avoid over-filtering)
        min_cells = max(1, min(min_cells, int(total_cells * 0.2)))

        # Log detailed information
        self.log(f"Data statistics:")
        self.log(f"  - Total cells: {total_cells}")
        self.log(f"  - Gene expression median: {median_expressing:.1f}")
        self.log(f"  - Gene expression Q25: {q25_expressing:.1f}")
        self.log(f"  - Gene expression Q75: {q75_expressing:.1f}")
        self.log(f"  - Distribution skewness: {distribution_skew:.3f}")
        self.log(f"  - Overall density: {overall_density:.3f}")

        self.log(f"Results from three methods:")
        self.log(f"  - Method 1 (quantile): {min_cells_method1}")
        self.log(f"  - Method 2 (significance): {min_cells_method2}")
        self.log(f"  - Method 3 (density): {min_cells_method3}")
        self.log(f"Final selection: {min_cells}")

        return min_cells
    
    def calculate_ppi_confidence_from_protein_abundance(self, adata_adt):
        """
        Calculate PPI network confidence threshold based on protein abundance

        Principle:
        - Interactions of high-abundance proteins are more likely to be detected
        - Determine PPI confidence threshold based on protein expression level distribution
        """
        self.log("Calculating PPI confidence threshold based on protein abundance...")

        # Get expression matrix
        X = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X

        # Calculate protein mean expression level
        protein_mean_expr = np.mean(X, axis=0)
        protein_var = np.var(X, axis=0)

        # Calculate coefficient of variation (CV = std/mean)
        cv = np.sqrt(protein_var) / (protein_mean_expr + 1e-6)

        # High-expression and low-variation proteins are more likely to have stable interactions
        # Determine confidence threshold based on expression level and stability
        high_expr_mask = protein_mean_expr > np.percentile(protein_mean_expr, 50)
        stable_mask = cv < np.percentile(cv, 50)
        reliable_proteins = high_expr_mask & stable_mask

        reliable_ratio = np.sum(reliable_proteins) / len(reliable_proteins)

        # Determine PPI confidence threshold based on reliable protein ratio
        if reliable_ratio > 0.5:  # Most proteins are reliable
            ppi_confidence = 0.8
        elif reliable_ratio > 0.3:  # Moderate reliability
            ppi_confidence = 0.7
        else:  # Low reliability, need stricter threshold
            ppi_confidence = 0.85

        self.log(f"Reliable protein ratio: {reliable_ratio:.3f}")
        self.log(f"PPI network confidence threshold: {ppi_confidence}")
        return ppi_confidence
    
    def estimate_impute_rate_from_dropout_pattern(self, adata_adt):
        """
        Estimate maximum imputation rate based on dropout pattern

        Principle:
        - Analyze dropout pattern in the data to avoid over-imputation
        - Determine reasonable imputation upper limit based on zero value distribution
        """
        self.log("Estimating maximum imputation rate based on dropout pattern...")

        # Get expression matrix
        X = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X

        # Calculate zero ratio
        zero_ratio = np.sum(X == 0) / X.size

        # Analyze dropout level per protein
        protein_dropout = np.sum(X == 0, axis=0) / X.shape[0]
        median_dropout = np.median(protein_dropout)

        # Determine maximum imputation rate based on dropout level
        # Higher dropout allows larger imputation rate, but with an upper limit
        if median_dropout > 0.7:  # High dropout
            max_impute_rate = min(0.1, median_dropout * 0.15)
        elif median_dropout > 0.5:  # Moderate dropout
            max_impute_rate = min(0.08, median_dropout * 0.12)
        else:  # Low dropout
            max_impute_rate = min(0.05, median_dropout * 0.1)

        # Ensure within reasonable range
        max_impute_rate = max(0.01, min(0.2, max_impute_rate))

        self.log(f"Data zero ratio: {zero_ratio:.3f}")
        self.log(f"Protein median dropout: {median_dropout:.3f}")
        self.log(f"Maximum imputation rate: {max_impute_rate:.3f}")
        return max_impute_rate
    
    def calculate_min_confidence_from_noise_level(self, adata_adt):
        """
        Calculate minimum imputation confidence based on noise level

        Principle:
        - Estimate noise level in the data
        - Set sufficiently high confidence threshold to avoid imputing noise
        """
        self.log("Calculating minimum imputation confidence based on noise level...")

        # Get expression matrix
        X = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X

        # Estimate technical noise level
        # Method: Based on variability of low-expression proteins
        protein_mean = np.mean(X, axis=0)
        protein_std = np.std(X, axis=0)

        # Variation in low-expression proteins mainly comes from technical noise
        low_expr_mask = protein_mean < np.percentile(protein_mean, 25)
        if np.sum(low_expr_mask) > 0:
            noise_level = np.median(protein_std[low_expr_mask] / (protein_mean[low_expr_mask] + 1e-6))
        else:
            noise_level = np.median(protein_std / (protein_mean + 1e-6))

        # Determine minimum confidence based on noise level
        if noise_level > 1.0:  # High noise
            min_confidence = 0.85
        elif noise_level > 0.5:  # Moderate noise
            min_confidence = 0.75
        else:  # Low noise
            min_confidence = 0.65

        # Ensure within reasonable range
        min_confidence = max(0.5, min(0.9, min_confidence))

        self.log(f"Estimated noise level: {noise_level:.3f}")
        self.log(f"Minimum imputation confidence: {min_confidence:.3f}")
        return min_confidence
    
    def estimate_top_percent_from_expression_variance(self, adata_adt):
        """
        Estimate imputation top N% positions based on expression variability

        Principle:
        - Higher variability indicates greater uncertainty in expression levels
        - Only impute positions with highest confidence
        """
        self.log("Estimating imputation percentage based on expression variability...")

        # Get expression matrix
        X = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X

        # Calculate expression variability
        protein_mean = np.mean(X, axis=0)
        protein_var = np.var(X, axis=0)

        # Calculate coefficient of variation
        cv = np.sqrt(protein_var) / (protein_mean + 1e-6)
        median_cv = np.median(cv)

        # Determine imputation percentage based on variability
        # Higher variability means impute fewer top positions only
        if median_cv > 2.0:  # High variability
            top_percent = 5
        elif median_cv > 1.0:  # Moderate variability
            top_percent = 10
        else:  # Low variability
            top_percent = 15

        # Additional consideration for data quality
        # If most proteins have low expression, reduce imputation rate
        low_expr_ratio = np.sum(protein_mean < np.percentile(protein_mean, 50)) / len(protein_mean)
        if low_expr_ratio > 0.7:
            top_percent = max(3, top_percent - 5)

        # Ensure within reasonable range
        top_percent = max(1, min(30, top_percent))

        self.log(f"Expression CV median: {median_cv:.3f}")
        self.log(f"Low expression protein ratio: {low_expr_ratio:.3f}")
        self.log(f"Imputation top percentage: {top_percent}%")
        return top_percent
    
    def calculate_atac_confidence_from_peak_tf_correlation(self, adata_rna, adata_atac):
        """
        Calculate ATAC imputation confidence threshold based on Peak-TF correlation

        Principle:
        - Analyze correlation between transcription factor gene expression and corresponding ATAC peaks
        - High correlation indicates reliability of Peak-TF mapping relationships
        """
        self.log("Calculating ATAC confidence threshold based on Peak-TF correlation...")

        # Get expression matrices
        X_rna = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X
        X_atac = adata_atac.X.toarray() if scipy.sparse.issparse(adata_atac.X) else adata_atac.X

        # If gene names and peak names overlap, calculate their correlations
        rna_genes = set(adata_rna.var_names)
        atac_peaks = set(adata_atac.var_names)

        # Find transcription factor genes (usually contain specific patterns in gene names)
        tf_genes = []
        for gene in rna_genes:
            # Common transcription factor gene patterns
            if any(pattern in gene.upper() for pattern in ['TF', 'FOX', 'SOX', 'HOX', 'PAX', 'MYC', 'MAX']):
                tf_genes.append(gene)

        if len(tf_genes) < 10:
            # If not enough TFs found, use highly variable genes as proxy
            if 'highly_variable' not in adata_rna.var.columns:
                sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=2000)
            tf_genes = adata_rna.var_names[adata_rna.var['highly_variable']][:200].tolist()

        # Calculate correlations between TF genes and ATAC peaks
        correlations = []
        for tf_gene in tf_genes[:100]:  # Limit computation
            if tf_gene in rna_genes:
                rna_idx = list(adata_rna.var_names).index(tf_gene)

                # Randomly sample some peaks to calculate correlations
                sample_peaks = np.random.choice(len(adata_atac.var_names),
                                              min(500, len(adata_atac.var_names)),
                                              replace=False)

                for peak_idx in sample_peaks:
                    corr, p_val = spearmanr(X_rna[:, rna_idx], X_atac[:, peak_idx])
                    if not np.isnan(corr) and p_val < 0.05:
                        correlations.append(abs(corr))

        if len(correlations) == 0:
            self.log("Warning: No significant Peak-TF correlations found, using default threshold")
            return 0.4

        correlations = np.array(correlations)

        # Determine confidence threshold based on correlation distribution
        # Use 70th percentile, slightly more lenient than GRN since Peak-TF relationships are more complex
        confidence_threshold = np.percentile(correlations, 70)

        # Ensure threshold is within reasonable range
        confidence_threshold = max(0.2, min(0.7, confidence_threshold))

        self.log(f"Calculated {len(correlations)} Peak-TF correlations")
        self.log(f"ATAC confidence threshold based on Peak-TF correlation: {confidence_threshold:.3f}")
        return confidence_threshold
    
    def estimate_atac_min_cells_from_accessibility_pattern(self, adata_atac):
        """
        Estimate ATAC minimum cells based on chromatin accessibility pattern

        Principle:
        - ATAC data is typically sparser than RNA data
        - Determine minimum cell threshold based on peak accessibility distribution
        """
        self.log("Estimating ATAC minimum cells based on accessibility pattern...")

        # Get expression matrix
        X = adata_atac.X.toarray() if scipy.sparse.issparse(adata_atac.X) else adata_atac.X

        # Calculate number of accessible cells per peak
        accessible_cells_per_peak = np.sum(X > 0, axis=0)

        # Analyze accessibility distribution
        median_accessible = np.median(accessible_cells_per_peak)
        q25_accessible = np.percentile(accessible_cells_per_peak, 25)
        q10_accessible = np.percentile(accessible_cells_per_peak, 10)

        # ATAC data is typically sparser than RNA, use smaller thresholds
        total_cells = X.shape[0]
        accessibility_ratio = median_accessible / total_cells

        if accessibility_ratio < 0.05:  # Extremely high sparsity
            min_cells = max(2, int(q10_accessible * 0.8))
        elif accessibility_ratio < 0.15:  # High sparsity
            min_cells = max(3, int(q25_accessible * 0.6))
        elif accessibility_ratio < 0.3:  # Moderate sparsity
            min_cells = max(5, int(median_accessible * 0.4))
        else:  # Low sparsity
            min_cells = max(8, int(median_accessible * 0.5))

        # Ensure minimum cells is within reasonable range
        min_cells = max(2, min(30, min_cells))

        self.log(f"Accessibility ratio: {accessibility_ratio:.3f}")
        self.log(f"Estimated ATAC minimum accessible cells: {min_cells}")
        return min_cells
    
    def biology_guided_parameter_selection(self, adata_rna, adata_adt=None, adata_atac=None):
        """
        Main function for automatic imputation parameter selection based on biological features

        Parameters:
            adata_rna: AnnData object for RNA data
            adata_adt: AnnData object for ADT/protein data (optional)
            adata_atac: AnnData object for ATAC data (optional)

        Returns:
            Dictionary containing available parameter sets
        """
        self.log("Starting biology-guided parameter selection...")

        results = {}

        # RNA parameter calculation
        self.log("\n=== RNA Parameter Calculation ===")
        results['rna_params'] = {
            'confidence_threshold': self.calculate_grn_confidence_from_co_expression(adata_rna),
            'min_cells': self.estimate_min_cells_from_gene_distribution(adata_rna)
        }

        # ADT parameter calculation (if ADT data provided)
        if adata_adt is not None:
            self.log("\n=== ADT Parameter Calculation ===")
            results['adt_params'] = {
                'confidence_threshold': self.calculate_ppi_confidence_from_protein_abundance(adata_adt),
                'max_impute_rate': self.estimate_impute_rate_from_dropout_pattern(adata_adt),
                'min_confidence_for_imputation': self.calculate_min_confidence_from_noise_level(adata_adt),
                'top_percent_to_impute': self.estimate_top_percent_from_expression_variance(adata_adt)
            }

        # ATAC parameter calculation (if ATAC data provided)
        if adata_atac is not None:
            self.log("\n=== ATAC Parameter Calculation ===")
            results['atac_params'] = {
                'confidence_threshold': self.calculate_atac_confidence_from_peak_tf_correlation(adata_rna, adata_atac),
                'min_cells': self.estimate_atac_min_cells_from_accessibility_pattern(adata_atac)
            }

        # Output final parameters
        self.log("\n=== Final Parameter Selection Results ===")
        for data_type, params in results.items():
            self.log(f"{data_type.upper()}:")
            for key, value in params.items():
                self.log(f"  {key}: {value}")

        return results
    
    def get_parameter_explanation(self, adata_rna, adata_adt=None, adata_atac=None):
        """
        Provide detailed explanation of parameter selection and data quality assessment
        """
        X_rna = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X

        report = {
            'data_quality': {
                'rna_sparsity': np.sum(X_rna == 0) / X_rna.size,
                'rna_genes': X_rna.shape[1],
                'cells': X_rna.shape[0]
            },
            'parameter_rationale': {
                'rna_confidence': "Based on 75th percentile of gene co-expression network correlation distribution",
                'rna_min_cells': "Based on sparsity analysis of gene expression distribution"
            }
        }

        if adata_adt is not None:
            X_adt = adata_adt.X.toarray() if scipy.sparse.issparse(adata_adt.X) else adata_adt.X
            report['data_quality'].update({
                'adt_sparsity': np.sum(X_adt == 0) / X_adt.size,
                'adt_proteins': X_adt.shape[1]
            })
            report['parameter_rationale'].update({
                'adt_confidence': "Based on comprehensive assessment of protein abundance and stability",
                'adt_max_rate': "Based on dropout pattern to avoid over-imputation",
                'adt_min_confidence': "Based on noise level to ensure imputation quality",
                'adt_top_percent': "Based on expression variability to ensure imputation accuracy"
            })

        if adata_atac is not None:
            X_atac = adata_atac.X.toarray() if scipy.sparse.issparse(adata_atac.X) else adata_atac.X
            report['data_quality'].update({
                'atac_sparsity': np.sum(X_atac == 0) / X_atac.size,
                'atac_peaks': X_atac.shape[1]
            })
            report['parameter_rationale'].update({
                'atac_confidence': "Based on 70th percentile of Peak-TF correlation distribution",
                'atac_min_cells': "Based on sparsity analysis of chromatin accessibility pattern"
            })

        return report

# Example usage function
def example_usage():
    """
    Usage example - Supports parameter selection for RNA, ADT and ATAC data
    """
    print("=== Biology-Guided Imputation Parameter Selector Usage Example ===\n")

    # Create parameter selector
    selector = BiologyGuidedParameterSelector(verbose=True)

    print("Supported data types and imputation methods:")
    print("1. RNA data - Using GRN network imputation")
    print("2. ADT data - Using PPI network imputation")
    print("3. ATAC data - Using Peak-TF mapping imputation\n")

    print("Usage example:")
    print("\n# Load data")
    print("# adata_rna = sc.read_h5ad('your_rna_data.h5ad')")
    print("# adata_adt = sc.read_h5ad('your_adt_data.h5ad')  # optional")
    print("# adata_atac = sc.read_h5ad('your_atac_data.h5ad')  # optional")

    print("\n# Get parameters (supports any combination)")
    print("# results = selector.biology_guided_parameter_selection(")
    print("#     adata_rna=adata_rna,")
    print("#     adata_adt=adata_adt,    # optional")
    print("#     adata_atac=adata_atac   # optional")
    print("# )")

    print("\n# Extract parameters for specific data types")
    print("# rna_params = results['rna_params']")
    print("# adt_params = results.get('adt_params', {})  # if no ADT data")
    print("# atac_params = results.get('atac_params', {})  # if no ATAC data")

    print("\n# Apply directly to imputation methods")
    print("# RNA imputation")
    print("# adata_rna_imputed = data_loader.impute_rna_with_grn(")
    print("#     adata_rna, adata_atac, **rna_params")
    print("# )")

    print("\n# ADT imputation (if ADT data available)")
    print("# if 'adt_params' in results:")
    print("#     adata_adt_imputed = data_loader.impute_protein_with_ppi(")
    print("#         adata_adt, **adt_params")
    print("#     )")

    print("\n# ATAC imputation (if ATAC data available)")
    print("# if 'atac_params' in results:")
    print("#     adata_atac_imputed = data_loader.impute_atac_with_peak_tf(")
    print("#         adata_rna, adata_atac, **atac_params")
    print("#     )")

    print("\n# Get detailed report")
    print("# report = selector.get_parameter_explanation(adata_rna, adata_adt, adata_atac)")

    print("\n=== ATAC Parameter Description ===")
    print("ATAC imputation method parameters:")
    print("- confidence_threshold: Peak-TF mapping confidence threshold (0.2-0.7)")
    print("- min_cells: Minimum number of cells where a peak must be open to consider imputation (2-30)")
    print("\nThese parameters are calculated based on the following biological principles:")
    print("1. Analyze correlation between transcription factor gene expression and ATAC peaks")
    print("2. Determine sparsity threshold based on chromatin accessibility pattern")
    print("3. Consider that ATAC data is typically sparser than RNA data")

if __name__ == "__main__":
    example_usage() 
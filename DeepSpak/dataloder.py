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
from .preprocess import construct_neighbor_graph, lsi, pca

from .protein_id_mapper import ProteinIDMapper
import random
# Set random seed for reproducibility
def setup_seed(seed=2020):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set global random seed
setup_seed(2020)


class SpatialOmicsDataLoader:
    """
    Spatial multi-omics data loader with GRN (Gene Regulatory Network) preprocessing

    Main features:
    1. Load RNA and ATAC data
    2. Use GRN to predict RNA and ATAC data
    3. Store raw data and predicted data in AnnData objects
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
        Initialize data loader

        Args:
            data_dir: Data directory
            trrust_file_path: TRRUST database file path
            peak_tf_db_path: Peak-TF database file path
            datatype: Data type, can be 'SPOTS' or 'Spatial-epigenome-transcriptome'
            force_reprocess: Whether to force reprocessing data
            cache_dir: Cache directory, defaults to cache subdirectory under data_dir
            string_protein_info_file: STRING database protein info file path for protein ID mapping
            ppi_file_path: PPI network file path for protein expression data imputation
            enable_protein_features: Whether to enable protein-related features, None for auto-detection
            enable_context_filtering: Whether to enable context gene filtering, default is True
        """
        self.data_dir = data_dir
        self.trrust_file_path = trrust_file_path
        self.peak_tf_db_path = peak_tf_db_path
        self.ppi_file_path = ppi_file_path
        self.datatype = datatype
        self.force_reprocess = force_reprocess
        self.cache_dir = cache_dir or os.path.join(data_dir, 'cache')
        self.enable_context_filtering = enable_context_filtering
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # TF-gene regulatory network
        self.tf_gene_network = None
        self.peak_tf_map = None
        
        
        self.string_protein_info_file = string_protein_info_file
        
        # Print protein feature related file status
        if ppi_file_path:
            print(f"PPI network file path: {ppi_file_path}")
        else:
            print("Warning: PPI network file path not set, protein imputation will be unavailable")

        if string_protein_info_file:
            print(f"STRING protein info file path: {string_protein_info_file}")
        else:
            print("Warning: STRING protein info file path not set, protein ID mapping will be limited")
        
        # TF-gene regulatory network lazy loading (requires adata_rna parameter)
        self.tf_gene_network = None
        self._tf_gene_network_loaded = False

        # If peak-TF database file is provided, load peak-TF mapping
        if self.peak_tf_db_path and os.path.exists(self.peak_tf_db_path):
            self.peak_tf_map = self._load_peak_tf_map()
    
    def _initialize_protein_id_mapper(self):
        """Initialize protein ID mapper and save to cache, only process ADT dataset related protein info"""
        # Initialize cache path
        if not hasattr(self, 'protein_mapping_cache'):
            self.protein_mapping_cache = os.path.join(self.cache_dir, 'protein_mapping.pkl')
            
        # First create ProteinIDMapper object
        print("Creating protein ID mapper...")
        self.protein_id_mapper = ProteinIDMapper(string_protein_info_file=self.string_protein_info_file)
        
        
        

        # Get ADT data related protein names - should get from actual ADT data, not just from manual mapping
        relevant_proteins = set()
        
       
        # STRING database info already loaded in protein_id_mapper initialization, just need to extract gene symbols
        if hasattr(self.protein_id_mapper, 'string_protein_info') and self.protein_id_mapper.string_protein_info:
            print(f"STRING database loaded in mapper: {len(self.protein_id_mapper.string_protein_info)} proteins")

                # Extract gene symbols to enhance mapping capability
            if not hasattr(self.protein_id_mapper, 'gene_symbol_to_string_ids'):
                self.protein_id_mapper.extract_gene_symbols_from_annotation()
        else:
            print("Warning: STRING database not loaded in mapper, mapping functionality may be limited")
        
        # Save protein ID mapper to cache
        self._save_protein_mapper_cache()

    def _save_protein_mapper_cache(self):
        """Save protein ID mapper to cache"""
        try:
            print(f"Saving protein ID mapper to cache: {self.protein_mapping_cache}")
            import pickle
            with open(self.protein_mapping_cache, 'wb') as f:
                pickle.dump(self.protein_id_mapper, f)
            print("Protein ID mapper saved to cache")
        except Exception as e:
            print(f"Failed to save protein ID mapper cache: {str(e)}")
    
    def _get_context_genes(self, adata, min_cells=5, min_expr=1.0):
        """Get context gene set that is actually expressed in data"""
        X = adata.raw.X if adata.raw is not None else adata.X

        if scipy.sparse.issparse(X):  # Sparse matrix
            gene_counts = (X > min_expr).sum(axis=0)
        else:  # Dense matrix
            gene_counts = (X > min_expr).sum(axis=0)

        # Force convert to 1D numpy array (solve matrix vs array issue)
        gene_counts = np.asarray(gene_counts).flatten()
        mask = gene_counts >= min_cells
        
        return set(adata.var_names[mask])
    
    def _ensure_tf_gene_network(self, adata_rna):
        """Ensure TF-gene network is loaded (lazy loading)"""
        if not self._tf_gene_network_loaded:
            if self.trrust_file_path and os.path.exists(self.trrust_file_path):
                print("First use, loading TF-gene regulatory network...")
                self.tf_gene_network = self._load_tf_gene_network(adata_rna)
                self._tf_gene_network_loaded = True
            else:
                print("Warning: TRRUST file not found, cannot load TF-gene regulatory network")
                self.tf_gene_network = None
                self._tf_gene_network_loaded = True
    
    def _load_tf_gene_network(self, adata_rna=None):
        """Load TF-gene regulatory network with context gene filtering"""
        print(f"Loading TF-gene regulatory network: {self.trrust_file_path}")
        tf_gene_dict = {}
        
        try:
            # Read TRRUST data
            # Format: TF  target  interaction_type  PMID
            trrust_df = pd.read_csv(self.trrust_file_path, sep='\t', header=None)

            if trrust_df.shape[1] >= 2:
                # Standardize column names, compatible with different formats
                if trrust_df.shape[1] >= 4:
                    trrust_df.columns = ['TF', 'Target', 'Mode', 'PMID']
                elif trrust_df.shape[1] == 3:
                    trrust_df.columns = ['TF', 'Target', 'Mode']
                else:
                    trrust_df.columns = ['TF', 'Target']
                
                print(f"Original TRRUST network: {len(trrust_df):,} edges, {trrust_df['TF'].nunique()} TFs")

                # Decide whether to use context gene filtering based on settings
                if self.enable_context_filtering and adata_rna is not None:
                    print("Applying context gene filtering...")

                    # Adaptively adjust parameters based on data type
                    n_cells = adata_rna.n_obs
                    if n_cells < 1000:  # Small dataset
                        min_cells, min_expr = 5, 0.5
                    elif n_cells < 5000:  # Medium dataset
                        min_cells, min_expr = 10, 0.5
                    else:  # Large dataset
                        min_cells, min_expr = 20, 1.0

                    print(f"  Dataset size: {n_cells} cells, using parameters: min_cells={min_cells}, min_expr={min_expr}")

                    # Get context genes
                    ctx_genes = self._get_context_genes(adata_rna, min_cells=min_cells, min_expr=min_expr)
                    print(f"  Context gene count: {len(ctx_genes):,}")

                    # Filter network: only keep edges where both TF and Target are in context genes
                    trrust_ctx = trrust_df[
                        trrust_df["TF"].isin(ctx_genes) & 
                        trrust_df["Target"].isin(ctx_genes)
                    ].reset_index(drop=True)
                    
                    print(f"  After context filtering: {len(trrust_ctx):,} edges (retained {len(trrust_ctx)/len(trrust_df):.1%})")

                    # Optional: remove super hub TFs
                    tf_degrees = trrust_ctx["TF"].value_counts()
                    max_degree = min(500, max(50, len(ctx_genes) // 20))  # Dynamically adjust hub threshold
                    hub_tfs = tf_degrees[tf_degrees > max_degree].index

                    if len(hub_tfs) > 0:
                        trrust_ctx = trrust_ctx[~trrust_ctx["TF"].isin(hub_tfs)]
                        print(f"  Removed {len(hub_tfs)} super hub TFs (degree > {max_degree})")
                        print(f"  Final network: {len(trrust_ctx):,} edges")

                    # Use filtered network
                    filtered_df = trrust_ctx
                elif adata_rna is not None:
                    print("Skipping context gene filtering (enable_context_filtering=False)")
                    filtered_df = trrust_df
                else:
                    print("Warning: adata_rna not provided, skipping context gene filtering")
                    filtered_df = trrust_df

                # Build dictionary format network
                for _, row in filtered_df.iterrows():
                    tf = row['TF']
                    target = row['Target']
                    
                    if tf not in tf_gene_dict:
                        tf_gene_dict[tf] = []
                    
                    if target not in tf_gene_dict[tf]:
                        tf_gene_dict[tf].append(target)
                
                print(f"Final TF-gene network: {len(tf_gene_dict)} TFs, {sum(len(genes) for genes in tf_gene_dict.values())} regulatory relationships")
                print(f"Context gene filtering status: {'enabled' if self.enable_context_filtering and adata_rna is not None else 'disabled'}")
                return tf_gene_dict
            else:
                print(f"TRRUST file format incorrect, column count: {trrust_df.shape[1]}")
                return None
        except Exception as e:
            print(f"Error loading TF-gene regulatory network: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_peak_tf_map(self):
        """Load peak-TF mapping relationships"""
        print(f"Loading peak-TF mapping: {self.peak_tf_db_path}")
        peak_tf_map = {}

        try:
            # Determine loading method based on file extension
            if self.peak_tf_db_path.endswith('.bed'):
                # BED format: chrom start end name score strand TFs
                df = pd.read_csv(self.peak_tf_db_path, sep='\t', header=None)

                if df.shape[1] >= 7:  # At least 7 columns needed
                    for _, row in df.iterrows():
                        peak_id = f"{row[0]}-{row[1]}-{row[2]}"  # chrom-start-end
                        tfs = row[6].split(',') if isinstance(row[6], str) else []
                        if tfs:
                            peak_tf_map[peak_id] = tfs
                else:
                    print(f"BED file format incorrect, column count: {df.shape[1]}")

            elif self.peak_tf_db_path.endswith('.csv'):
                # CSV format: peak_id,tf1,tf2,...
                df = pd.read_csv(self.peak_tf_db_path)

                if 'peak_id' in df.columns and 'tfs' in df.columns:
                    for _, row in df.iterrows():
                        peak_id = row['peak_id']
                        tfs = row['tfs'].split(',') if isinstance(row['tfs'], str) else []
                        if tfs:
                            peak_tf_map[peak_id] = tfs
                else:
                    print(f"CSV file missing required columns: peak_id or tfs")

            elif self.peak_tf_db_path.endswith('.json'):
                # JSON format: {"peak_id": ["tf1", "tf2", ...]}
                with open(self.peak_tf_db_path, 'r') as f:
                    peak_tf_map = json.load(f)

            else:
                print(f"Unsupported file format: {self.peak_tf_db_path}")
                return None

            print(f"Successfully loaded peak-TF mapping: {len(peak_tf_map)} peaks, {sum(len(tfs) for tfs in peak_tf_map.values())} associations")
            return peak_tf_map

        except Exception as e:
            print(f"Error loading peak-TF mapping: {str(e)}")
            return None
    
    

    def impute_rna_with_grn(self, adata_omics1, adata_omics2, confidence_threshold=0.3, min_cells=5):

        """
        Use GRN predicted data to impute raw RNA data matrix

        Args:
            adata_omics1: AnnData object for RNA data
            adata_omics2: AnnData object for ATAC data
            confidence_threshold: Prediction confidence threshold, predictions below this will be ignored
            min_cells: Minimum number of cells a gene must be expressed in to be considered for imputation
        """
        print("Using GRN predicted data to impute raw RNA data...")

        # Ensure TF-gene network is loaded (with context gene filtering)
        self._ensure_tf_gene_network(adata_omics1)

        if self.tf_gene_network is None:
            print("Warning: TF-gene regulatory network not loaded, cannot perform imputation")
            return adata_omics1

        # Get original expression matrix
        X_orig = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X

        # Create gene to index mapping
        gene_to_idx = {gene: i for i, gene in enumerate(adata_omics1.var_names)}

        # Find all predictable genes (genes in GRN network)
        predictable_genes = set()
        # Use sorted dictionary keys to ensure consistent traversal order
        for tf in sorted(self.tf_gene_network.keys()):
            targets = self.tf_gene_network[tf]
            if tf in gene_to_idx:  # Ensure TF is in dataset
                for target in targets:
                    if target in gene_to_idx:  # Ensure target gene is in dataset
                        predictable_genes.add(target)

        print(f"Found {len(predictable_genes)} predictable genes in GRN network")

        # Calculate expression frequency for each gene
        gene_expression_freq = np.sum(X_orig > 0, axis=0)

        # Find genes to impute (predictable genes with expression frequency below min_cells)
        # Use sorting to ensure consistent order
        genes_to_impute = []
        for gene in sorted(predictable_genes):
            gene_idx = gene_to_idx[gene]
            if gene_expression_freq[gene_idx] < min_cells:
                genes_to_impute.append(gene_idx)

        print(f"Found {len(genes_to_impute)} genes to impute")

        # Process each gene to impute
        filled_genes = 0
        filled_values = 0

        for gene_idx in tqdm(genes_to_impute, desc="Imputing gene expression"):
            gene_name = adata_omics1.var_names[gene_idx]

            # Find TFs regulating this gene
            regulating_tfs = []
            # Use sorted dictionary keys to ensure consistent traversal order
            for tf in sorted(self.tf_gene_network.keys()):
                targets = self.tf_gene_network[tf]
                if gene_name in targets and tf in gene_to_idx:
                    regulating_tfs.append(tf)

            if not regulating_tfs:
                continue

            # Calculate average TF expression as predicted value
            tf_expr_sum = np.zeros(X_orig.shape[0])
            # Use sorted TF list to ensure consistent calculation order
            for tf in sorted(regulating_tfs):
                tf_idx = gene_to_idx[tf]
                tf_expr_sum += X_orig[:, tf_idx]

            # Calculate average TF expression
            predicted_expr = tf_expr_sum / len(regulating_tfs)

            # Apply confidence threshold
            predicted_expr[predicted_expr < confidence_threshold] = 0

            # Only impute positions that are zero in original data
            original_zero_mask = X_orig[:, gene_idx] == 0
            changes = np.sum(original_zero_mask & (predicted_expr > 0))

            if changes > 0:
                X_orig[original_zero_mask, gene_idx] = predicted_expr[original_zero_mask]
                filled_genes += 1
                filled_values += changes

        # Update original data matrix
        adata_omics1.X = scipy.sparse.csr_matrix(X_orig)

        # Calculate imputation effect
        print(f"Imputation statistics:")
        print(f"- Successfully imputed {filled_genes} genes")
        print(f"- Total {filled_values} values imputed")
        print(f"- Non-zero element ratio after imputation: {np.count_nonzero(X_orig) / X_orig.size:.2%}")

        return adata_omics1

    def evaluate_imputation_quality(self, X_orig, X_imputed):
        """Evaluate imputation quality"""
        # Calculate non-zero ratio before and after imputation
        original_nonzero = np.count_nonzero(X_orig) / X_orig.size
        imputed_nonzero = np.count_nonzero(X_imputed) / X_imputed.size

        # Calculate gene expression distribution before and after imputation
        original_means = np.mean(X_orig, axis=0)
        imputed_means = np.mean(X_imputed, axis=0)

        # Calculate correlation before and after imputation
        correlation = np.corrcoef(original_means, imputed_means)[0,1]

        print(f"Imputation statistics:")
        print(f"- Non-zero ratio: {original_nonzero:.2%} -> {imputed_nonzero:.2%}")
        print(f"- Gene expression correlation: {correlation:.4f}")

    def impute_atac_with_peak_tf(self, adata_omics1, adata_omics2, confidence_threshold=0.3, min_cells=5):
        """Impute ATAC data using peak-TF mapping relationships"""
        print("Imputing ATAC data using peak-TF mapping...")

        if self.peak_tf_map is None:
            print("Warning: Peak-TF mapping not loaded, cannot perform imputation")
            return adata_omics2

        # Get original ATAC matrix
        X_atac = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else adata_omics2.X

        # Get RNA expression matrix
        X_rna = adata_omics1.X.toarray() if scipy.sparse.issparse(adata_omics1.X) else adata_omics1.X

        # Create gene to index mapping
        gene_to_idx = {gene: i for i, gene in enumerate(adata_omics1.var_names)}

        # Create peak to index mapping (also handle different naming formats)
        peak_to_idx = {}
        for i, peak in enumerate(adata_omics2.var_names):
            # Save original format
            peak_to_idx[peak] = i

            # Handle possible format differences
            if ':' in peak:
                # Convert "chr1:1000-2000" to "chr1-1000-2000"
                parts = peak.split(':')
                if len(parts) == 2 and '-' in parts[1]:
                    chrom = parts[0]
                    pos = parts[1].split('-')
                    if len(pos) == 2:
                        alt_peak = f"{chrom}-{pos[0]}-{pos[1]}"
                        peak_to_idx[alt_peak] = i
            elif '-' in peak:
                # Convert "chr1-1000-2000" to "chr1:1000-2000"
                parts = peak.split('-')
                if len(parts) >= 3:
                    chrom, start, end = parts[0], parts[1], parts[2]
                    alt_peak = f"{chrom}:{start}-{end}"
                    peak_to_idx[alt_peak] = i

        print(f"Total ATAC peaks: {len(adata_omics2.var_names)}")
        print(f"Peak mapping count after format processing: {len(peak_to_idx)}")

        # Find predictable peaks (map peaks to indices)
        predictable_peaks_idx = set()
        peak_tf_matches = 0

        # Use sorted dictionary keys to ensure consistent traversal order
        for peak in sorted(self.peak_tf_map.keys()):
            tfs = self.peak_tf_map[peak]
            if peak in peak_to_idx:
                peak_tf_matches += 1
                predictable_peaks_idx.add(peak_to_idx[peak])

        print(f"Found {peak_tf_matches} matching peaks in peak-TF mapping, predictable peak index count: {len(predictable_peaks_idx)}")

        # Directly impute sparse regions, not relying on global min_cells threshold
        zeros_count = np.sum(X_atac == 0)
        print(f"Zero count in ATAC data: {zeros_count}, ratio: {zeros_count/X_atac.size:.2%}")

        # Process each predictable peak, count possible imputations
        filled_peaks = 0
        filled_values = 0

        # First count zero-value peaks per cell
        zero_peaks_per_cell = np.sum(X_atac == 0, axis=1)
        print(f"Average zero-value peaks per cell: {np.mean(zero_peaks_per_cell):.1f}")

        # Impute predictable peaks, use sorting to ensure consistent order
        for peak_idx in tqdm(sorted(predictable_peaks_idx), desc="Imputing peak accessibility"):
            # Get peak name
            peak_name = adata_omics2.var_names[peak_idx]

            # Find TFs regulating this peak
            regulating_tfs = []
            # Use sorted dictionary keys to ensure consistent traversal order
            for peak in sorted(self.peak_tf_map.keys()):
                tfs = self.peak_tf_map[peak]
                if peak in peak_to_idx and peak_to_idx[peak] == peak_idx:
                    regulating_tfs = tfs
                    break

            if not regulating_tfs:
                continue

            # Count regulatory TFs in RNA data, use sorting to ensure consistent order
            valid_tfs = sorted([tf for tf in regulating_tfs if tf in gene_to_idx])
            if not valid_tfs:
                continue

            # Get zero-value cell positions
            original_zero_mask = X_atac[:, peak_idx] == 0

            # If peak has no zeros, skip processing
            if np.sum(original_zero_mask) == 0:
                continue

            # Calculate average TF expression as predicted value
            tf_expr_values = np.array([X_rna[:, gene_to_idx[tf]] for tf in valid_tfs])
            predicted_expr = np.mean(tf_expr_values, axis=0)

            # Apply confidence threshold
            predicted_expr[predicted_expr < confidence_threshold] = 0

            # Only impute positions that are zero in original data
            changes = np.sum(original_zero_mask & (predicted_expr > 0))

            if changes > 0:
                X_atac[original_zero_mask, peak_idx] = predicted_expr[original_zero_mask]
                filled_peaks += 1
                filled_values += changes

        # Update ATAC data matrix
        adata_omics2.X = scipy.sparse.csr_matrix(X_atac)

        # Calculate imputation effect
        print(f"Imputation statistics:")
        print(f"- Successfully imputed {filled_peaks} peaks")
        print(f"- Total {filled_values} values imputed")
        print(f"- Non-zero element ratio before imputation: {(X_atac.size - zeros_count) / X_atac.size:.2%}")
        print(f"- Non-zero element ratio after imputation: {np.count_nonzero(X_atac) / X_atac.size:.2%}")

        return adata_omics2

    def load_ppi_network(self):
        """Load STRING format PPI network file, prioritize loading from cache, integrate protein ID mapper"""

        # Initialize necessary attributes (if not already initialized)
        if not hasattr(self, 'ppi_network_cache'):
            self.ppi_network_cache = os.path.join(self.cache_dir, 'ppi_network.pkl')
        if not hasattr(self, 'protein_mapping_cache'):
            self.protein_mapping_cache = os.path.join(self.cache_dir, 'protein_mapping.pkl')
        if not hasattr(self, 'ppi_network'):
            self.ppi_network = None

        # Initialize protein ID mapper
        if not hasattr(self, 'protein_id_mapper') or self.protein_id_mapper is None:
            self._initialize_protein_id_mapper()

        # If PPI network already loaded, return directly
        if self.ppi_network is not None:
            print(f"Using already loaded PPI network")
            return self.ppi_network

        # Try to load from cache
        if not self.force_reprocess and os.path.exists(self.ppi_network_cache):
            try:
                print(f"Loading PPI network from cache: {self.ppi_network_cache}")
                import pickle
                with open(self.ppi_network_cache, 'rb') as f:
                    self.ppi_network = pickle.load(f)

                # Validate loaded network
                if self.ppi_network and isinstance(self.ppi_network, dict):
                    num_proteins = len(self.ppi_network)
                    num_interactions = sum(len(interactions) for interactions in self.ppi_network.values()) // 2
                    print(f"Successfully loaded PPI network from cache: {num_proteins} proteins, {num_interactions} interactions")
                    return self.ppi_network
                else:
                    print("PPI network in cache is invalid, will reload")
                    self.ppi_network = None
            except Exception as e:
                print(f"Failed to load PPI network cache: {str(e)}")
                self.ppi_network = None

        # If cache loading failed or force reprocess, load from file
        print(f"Loading PPI network from file: {self.ppi_file_path}")

        try:
            # First check if PPI file path is valid
            if self.ppi_file_path is None:
                print(f"Error: PPI file path not set")
                return None

            # Check if file exists
            if not os.path.exists(self.ppi_file_path):
                print(f"Error: PPI file does not exist: {self.ppi_file_path}")
                return None

            # Read first few lines to determine format
            with open(self.ppi_file_path, 'r') as f:
                first_lines = [next(f) for _ in range(5) if f]

            # Check file format
            if len(first_lines) == 0:
                print(f"Error: PPI file is empty")
                return None

            # Determine delimiter
            delimiter = '\t' if '\t' in first_lines[0] else ' '
            print(f"Using delimiter: '{delimiter}'")

            # Check if there's a header row
            has_header = False
            if any(x in first_lines[0].lower() for x in ['protein', 'score', 'combined']):
                has_header = True
                print("Header row detected")
            
            # Get ADT data related protein IDs
            relevant_protein_ids = set()
            if self.protein_id_mapper:
                print("Using protein ID mapper to get related STRING IDs...")

                # 1. Get all STRING IDs from manual mapping (now empty, mainly rely on STRING database)
                manual_count = 0
                for protein_name, mapping_data in self.protein_id_mapper.manual_mapping.items():
                    if isinstance(mapping_data, list) and len(mapping_data) >= 2:
                        # Compatible with original format: [gene ID, [protein ID list]]
                        protein_ids = mapping_data[1] if isinstance(mapping_data[1], list) else [mapping_data[1]]
                        for protein_id in protein_ids:
                            relevant_protein_ids.add(protein_id)
                            manual_count += 1
                print(f"Obtained {manual_count} protein IDs from manual mapping")

                # 2. Get from gene symbol mapping
                if hasattr(self.protein_id_mapper, 'gene_symbol_to_string_ids'):
                    gene_symbol_count = 0
                    for string_ids in self.protein_id_mapper.gene_symbol_to_string_ids.values():
                        relevant_protein_ids.update(string_ids)
                        gene_symbol_count += len(string_ids)
                    print(f"Obtained {gene_symbol_count} STRING IDs from gene symbol mapping")

                # 3. Get from preferred name mapping
                if hasattr(self.protein_id_mapper, 'preferred_name_to_string_id'):
                    preferred_count = 0
                    for string_ids in self.protein_id_mapper.preferred_name_to_string_id.values():
                        relevant_protein_ids.update(string_ids)
                        preferred_count += len(string_ids)
                    print(f"Obtained {preferred_count} STRING IDs from preferred name mapping")

                print(f"Obtained total {len(relevant_protein_ids)} unique protein IDs from mapper")
            else:
                print("Warning: Protein ID mapper not initialized")


            print(f"Found {len(relevant_protein_ids)} protein IDs related to ADT data")

            # Read STRING format PPI data, only load data related to relevant proteins
            print("Starting to read PPI data...")

            # Use chunked reading to reduce memory usage
            chunk_size = 100000  # Number of rows to read at a time
            ppi_network = {}
            total_lines = 0
            relevant_lines = 0

            # Determine column names
            if has_header:
                df_header = pd.read_csv(self.ppi_file_path, sep=delimiter, nrows=0)
                protein1_col = next((col for col in df_header.columns if 'protein1' in str(col).lower()), None)
                protein2_col = next((col for col in df_header.columns if 'protein2' in str(col).lower()), None)
                score_col = next((col for col in df_header.columns if 'score' in str(col).lower() or 'combined' in str(col).lower()), None)

                if not (protein1_col and protein2_col and score_col):
                    print(f"Warning: Cannot recognize column names, columns: {list(df_header.columns)}")
                    protein1_col = df_header.columns[0]
                    protein2_col = df_header.columns[1]
                    score_col = df_header.columns[2]
            else:
                # Assume first three columns are protein1, protein2, score
                protein1_col = 0
                protein2_col = 1
                score_col = 2

            print(f"Using columns: protein1={protein1_col}, protein2={protein2_col}, score={score_col}")

            # Read file in chunks
            for chunk in pd.read_csv(self.ppi_file_path, sep=delimiter, chunksize=chunk_size, header=0 if has_header else None):
                total_lines += len(chunk)

                # Filter rows containing relevant proteins
                if isinstance(protein1_col, int):
                    # Use column index
                    relevant_chunk = chunk[(chunk.iloc[:, protein1_col].isin(relevant_protein_ids)) |
                                         (chunk.iloc[:, protein2_col].isin(relevant_protein_ids))]
                else:
                    # Use column names
                    relevant_chunk = chunk[(chunk[protein1_col].isin(relevant_protein_ids)) |
                                         (chunk[protein2_col].isin(relevant_protein_ids))]

                relevant_lines += len(relevant_chunk)

                # Process relevant rows
                for _, row in relevant_chunk.iterrows():
                    p1 = str(row[protein1_col])
                    p2 = str(row[protein2_col])

                    # Ensure protein IDs have entries in network
                    if p1 not in ppi_network:
                        ppi_network[p1] = {}
                    if p2 not in ppi_network:
                        ppi_network[p2] = {}

                    # Ensure score is numeric
                    try:
                        conf = float(row[score_col])
                        # If score range is 0-1000, normalize to 0-1
                        if conf > 1:
                            conf = conf / 1000.0
                    except:
                        # If conversion fails, use default value
                        conf = 0.5

                    # Add bidirectional relationship
                    ppi_network[p1][p2] = conf
                    ppi_network[p2][p1] = conf

                # Print progress
                if total_lines % (chunk_size * 10) == 0:
                    print(f"Processed {total_lines} lines, found {relevant_lines} relevant lines")

            # Check network build result
            if not ppi_network:
                print("Error: Failed to build PPI network")
                return None

            # Network statistics
            num_proteins = len(ppi_network)
            num_interactions = sum(len(interactions) for interactions in ppi_network.values()) // 2  # Divide by 2 because each interaction is counted twice

            print(f"Successfully loaded PPI network: {num_proteins} proteins, {num_interactions} interactions")
            print(f"Filtered {relevant_lines} relevant lines from {total_lines} original lines")
            
            # Check if our manual mapping IDs are in network
            test_ids = [
                'ENSP00000359722',  # CD163
                'ENSP00000367890',  # CR2
                'ENSP00000368438'   # PCNA
            ]

            print("\nChecking if manual mapping IDs are in PPI network:")
            for test_id in test_ids:
                if test_id in ppi_network:
                    num_interactions = len(ppi_network[test_id])
                    print(f"  {test_id} in network, has {num_interactions} interactions")
                    # Print first 3 interaction partners
                    partners = list(ppi_network[test_id].keys())[:3]
                    print(f"    Interaction partner examples: {partners}")
                else:
                    print(f"  {test_id} not in network")

            # Save PPI network to cache
            try:
                print(f"Saving PPI network to cache: {self.ppi_network_cache}")
                import pickle
                with open(self.ppi_network_cache, 'wb') as f:
                    pickle.dump(ppi_network, f)
                print("PPI network saved to cache")
            except Exception as e:
                print(f"Failed to save PPI network cache: {str(e)}")

            # Save loaded network to instance variable
            self.ppi_network = ppi_network
            return ppi_network
        except Exception as e:
            print(f"Error loading PPI network: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def impute_protein_with_ppi(self, adata, confidence_threshold=0.7, use_rna_data=False, rna_adata=None, max_impute_rate=0.05, min_confidence_for_imputation=0.8, top_percent_to_impute=10):
        """
        Impute protein expression data using PPI network

        Args:
            adata: AnnData object containing protein expression data
            confidence_threshold: PPI confidence threshold, only use interactions above this value
            use_rna_data: Whether to use RNA data for imputation
            rna_adata: AnnData object for RNA data, used when use_rna_data=True
            max_impute_rate: Maximum imputation rate, default is 0.05 (5%)
            min_confidence_for_imputation: Minimum confidence threshold for imputation, default is 0.8
            top_percent_to_impute: Only impute positions with predicted values in top N%, default is 10%

        Returns:
            Updated AnnData object
        """
        # Ensure protein ID mapper is initialized
        if not hasattr(self, 'protein_id_mapper') or self.protein_id_mapper is None:
            self._initialize_protein_id_mapper()

        print("Imputing protein expression data using PPI network...")
        print(f"Using confidence threshold: {confidence_threshold}")
        print(f"Maximum imputation rate: {max_impute_rate*100}%")
        print(f"Minimum confidence threshold for imputation: {min_confidence_for_imputation}")
        print(f"Only imputing top {top_percent_to_impute}% positions by predicted value")

        # Load PPI network
        ppi_network = self.load_ppi_network()
        if ppi_network is None:
            print("Warning: PPI network not loaded, cannot perform imputation")
            return adata

        # Get original expression matrix
        X_orig = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

        # Create protein to index mapping
        protein_to_idx = {protein: i for i, protein in enumerate(adata.var_names)}
        print(f"Number of proteins in ADT data: {len(protein_to_idx)}")
        print(f"First 10 protein names: {list(adata.var_names[:10])}")

        # Check if STRING ID mapping already exists
        has_string_ids = 'string_ids' in adata.var.columns
        if has_string_ids:
            print("Detected existing STRING ID mapping, will use existing mapping")

        # Create STRING ID to ADT protein mapping
        string_to_adt = {}
        adt_found_in_ppi = set()

        if has_string_ids:
            # Use existing STRING ID mapping
            for i, protein in enumerate(adata.var_names):
                string_ids_str = adata.var['string_ids'].iloc[i]  # Use .iloc to avoid FutureWarning
                if string_ids_str:  # If mapping exists
                    # Split string to get all STRING IDs
                    string_ids = string_ids_str.split(',')

                    # Check if each STRING ID is in PPI network
                    for string_id in string_ids:
                        if string_id in ppi_network:
                            string_to_adt[string_id] = protein
                            adt_found_in_ppi.add(protein)

            print(f"Proteins found using existing mapping: {len(adt_found_in_ppi)}")
            if len(adt_found_in_ppi) > 0:
                print(f"Mapped protein examples: {list(adt_found_in_ppi)[:5]}")

            # Print first 10 STRING IDs and whether they are in PPI network
            print("\nChecking if first 10 proteins' STRING IDs are in PPI network:")
            for i in range(min(10, adata.n_vars)):
                protein = adata.var_names[i]
                string_ids_str = adata.var['string_ids'].iloc[i]
                if string_ids_str:
                    string_ids = string_ids_str.split(',')

                    in_network = [id for id in string_ids if id in ppi_network]
                    print(f"  {protein}: STRING IDs={string_ids}, in network={in_network}")
                else:
                    print(f"  {protein}: No STRING ID mapping")

        # If not enough proteins found, try other matching methods
        if len(adt_found_in_ppi) < len(protein_to_idx) * 0.1:  # If match rate below 10%
            print("Existing mapping match rate too low, trying other matching methods...")

            # 1. Direct matching
            direct_matched = set()
            for protein in adata.var_names:
                if protein not in adt_found_in_ppi:
                    if protein in ppi_network:
                        string_to_adt[protein] = protein
                        adt_found_in_ppi.add(protein)
                        direct_matched.add(protein)

            print(f"Direct matched proteins: {len(direct_matched)}")
            if len(direct_matched) > 0:
                print(f"Direct matched protein examples: {list(direct_matched)[:5]}")

            # 2. Try matching after removing prefix
            prefix_matched = set()
            for protein in adata.var_names:
                if protein not in adt_found_in_ppi:
                    # If protein name contains underscore or dot, try splitting
                    parts = protein.split('_')
                    if len(parts) > 1:
                        base_name = parts[-1]  # Take last part
                        if base_name in ppi_network:
                            string_to_adt[base_name] = protein
                            adt_found_in_ppi.add(protein)
                            prefix_matched.add(protein)

            print(f"Matched proteins after removing prefix: {len(prefix_matched)}")
            if len(prefix_matched) > 0:
                print(f"Prefix matched protein examples: {list(prefix_matched)[:5]}")

            # 3. Fuzzy matching - check if STRING ID contains ADT protein name
            fuzzy_matched = set()
            for protein in adata.var_names:
                if protein not in adt_found_in_ppi:
                    for string_id in ppi_network.keys():
                        # Check if STRING ID contains protein name
                        if protein.lower() in string_id.lower():
                            string_to_adt[string_id] = protein
                            adt_found_in_ppi.add(protein)
                            fuzzy_matched.add(protein)
                            break

            print(f"Fuzzy matched protein count: {len(fuzzy_matched)}")
            if len(fuzzy_matched) > 0:
                print(f"Fuzzy matched protein examples: {list(fuzzy_matched)[:5]}")

            # 5. Use manual mapping
            manual_matched = set()
            if hasattr(self, 'protein_id_mapper') and self.protein_id_mapper:
                print("Using protein ID mapper for matching...")
                for protein in adata.var_names:
                    if protein not in adt_found_in_ppi:
                        # Use mapper to get STRING ID
                        string_ids = self.protein_id_mapper.map_protein_to_string_ids(protein)
                        for string_id in string_ids:
                            if string_id in ppi_network:
                                string_to_adt[string_id] = protein
                                adt_found_in_ppi.add(protein)
                                manual_matched.add(protein)
                                break

                print(f"Matched proteins using mapper: {len(manual_matched)}")
                if len(manual_matched) > 0:
                    print(f"Mapper matched protein examples: {list(manual_matched)[:5]}")

            # 6. More lenient fuzzy matching - check partial string
            loose_matched = set()
            if len(adt_found_in_ppi) < len(protein_to_idx) * 0.1:  # If match rate below 10%
                print("Warning: Match rate too low, trying more lenient matching...")

                for protein in adata.var_names:
                    if protein not in adt_found_in_ppi:
                        # If protein name length > 3, try matching partial string
                        if len(protein) > 3:
                            for string_id in ppi_network.keys():
                                # Check if STRING ID contains first 3 characters of protein name
                                if protein[:3].lower() in string_id.lower():
                                    string_to_adt[string_id] = protein
                                    adt_found_in_ppi.add(protein)
                                    loose_matched.add(protein)
                                    break

            print(f"Lenient matched protein count: {len(loose_matched)}")
            if len(loose_matched) > 0:
                print(f"Lenient matched protein examples: {list(loose_matched)[:5]}")

        print(f"Total matched proteins: {len(adt_found_in_ppi)}")
        print(f"Match rate: {len(adt_found_in_ppi) / len(protein_to_idx):.2%}")

        # Output all matched protein names and corresponding STRING IDs
        print("\nMatched proteins and STRING ID correspondence:")
        # Reverse mapping, find STRING IDs for each ADT protein
        adt_to_string_ids = {}
        for string_id, adt_protein in string_to_adt.items():
            if adt_protein not in adt_to_string_ids:
                adt_to_string_ids[adt_protein] = []
            adt_to_string_ids[adt_protein].append(string_id)

        # Output sorted by ADT protein name
        sorted_adt_proteins = sorted(adt_to_string_ids.keys())
        for i, adt_protein in enumerate(sorted_adt_proteins):
            string_ids = adt_to_string_ids[adt_protein]
            # Limit output count to avoid too much output
            if i < 20:  # Only output first 20
                print(f"  ADT protein: {adt_protein} -> STRING ID: {string_ids[:3]}{'...' if len(string_ids) > 3 else ''}")

        if len(sorted_adt_proteins) > 20:
            print(f"  ... {len(sorted_adt_proteins) - 20} more matches not shown")

        # If no proteins matched, cannot perform imputation
        if not string_to_adt:
            print("Error: No proteins matched, cannot perform imputation")
            return adata

        # Reverse mapping: ADT protein -> STRING ID list
        adt_to_string = {}
        for string_id, adt_protein in string_to_adt.items():
            if adt_protein not in adt_to_string:
                adt_to_string[adt_protein] = []
            adt_to_string[adt_protein].append(string_id)

        # Find all predictable proteins (proteins in PPI network)
        predictable_proteins = set()
        # Use sorting to ensure consistent traversal order
        for adt_protein in sorted(adt_to_string.keys()):
            string_ids = adt_to_string[adt_protein]
            for string_id in sorted(string_ids):
                if string_id in ppi_network:
                    for p2 in sorted(ppi_network[string_id].keys()):
                        if p2 in string_to_adt:
                            predictable_proteins.add(string_to_adt[p2])

        print(f"Found {len(predictable_proteins)} predictable proteins in PPI network")
        # Display after sorting to ensure output consistency
        sorted_predictable = sorted(list(predictable_proteins))
        print(f"Predictable protein examples: {sorted_predictable[:10]}")

        # Calculate expression frequency for each protein
        protein_expression_freq = np.sum(X_orig > 0, axis=0)

        # Find proteins to impute (predictable proteins with lower expression frequency)
        proteins_to_impute = []
        # Use sorting to ensure consistent processing order
        for protein in sorted(predictable_proteins):
            protein_idx = protein_to_idx[protein]
            if protein_expression_freq[protein_idx] < np.mean(protein_expression_freq):
                proteins_to_impute.append(protein_idx)

        print(f"Found {len(proteins_to_impute)} proteins to impute")
        print(f"Proteins to impute examples: {[adata.var_names[idx] for idx in proteins_to_impute[:10]]}")

        # Process each protein to impute
        filled_proteins = 0
        filled_values = 0
        filled_protein_names = []

        # If using RNA data, need to establish gene name to index mapping
        gene_to_idx = {}
        if use_rna_data and rna_adata is not None:
            gene_to_idx = {gene: i for i, gene in enumerate(rna_adata.var_names)}
            print(f"Number of genes in RNA data: {len(gene_to_idx)}")

        # Calculate zero count for each protein, used to control imputation rate
        protein_zeros = np.sum(X_orig == 0, axis=0)

        # Create dictionary to track imputation rates
        protein_impute_rates = {}

        # Total samples per protein
        total_samples = X_orig.shape[0]

        for protein_idx in tqdm(proteins_to_impute, desc="Imputing protein expression"):
            protein_name = adata.var_names[protein_idx]

            # Find proteins interacting with this protein
            interacting_proteins = []
            interaction_weights = []

            # Get STRING IDs for this protein
            string_ids = adt_to_string.get(protein_name, [])

            # Add debug info (output after sorting to ensure consistency)
            print(f"Processing protein: {protein_name}, STRING IDs: {sorted(string_ids)}")

            # Sort STRING IDs to ensure consistent processing order
            for string_id in sorted(string_ids):
                if string_id in ppi_network:
                    print(f"  {string_id} in PPI network, has {len(ppi_network[string_id])} interactions")

                    # 1. First try to find interacting proteins in ADT data
                    adt_interactors = 0
                    # Sort interactions in PPI network to ensure consistent processing order
                    for p2 in sorted(ppi_network[string_id].keys()):
                        conf = ppi_network[string_id][p2]
                        if conf >= confidence_threshold and p2 in string_to_adt:
                            interacting_proteins.append(string_to_adt[p2])
                            interaction_weights.append(conf)
                            adt_interactors += 1

                    print(f"  Found {adt_interactors} interacting proteins in ADT data (confidence threshold >= {confidence_threshold})")

                    # 2. If not enough interacting proteins found in ADT, try using RNA data
                    if adt_interactors == 0 and use_rna_data and rna_adata is not None:
                        rna_interactors = 0
                        # Sort interactions in PPI network to ensure consistent processing order
                        for p2 in sorted(ppi_network[string_id].keys()):
                            conf = ppi_network[string_id][p2]
                            if conf >= confidence_threshold:
                                # Extract gene symbol from STRING ID
                                gene_symbol = None
                                if '.' in p2:  # If ID is like "ENSP00000350844" or "9606.ENSP00000350844"
                                    if hasattr(self, 'protein_id_mapper') and self.protein_id_mapper:
                                        gene_symbol = self.protein_id_mapper.map_string_id_to_protein(p2)

                                # Check if gene symbol is in RNA data
                                if gene_symbol and gene_symbol in gene_to_idx:
                                    # Use expression value from RNA data
                                    gene_idx = gene_to_idx[gene_symbol]
                                    rna_expr = rna_adata.X[:, gene_idx].toarray().flatten() if scipy.sparse.issparse(rna_adata.X) else rna_adata.X[:, gene_idx]

                                    # Only consider if RNA expression is not all zeros
                                    if np.any(rna_expr > 0):
                                        # Create a virtual "ADT protein" identifier
                                        virtual_protein = f"RNA_{gene_symbol}"
                                        # Add to interaction list
                                        interacting_proteins.append(virtual_protein)
                                        interaction_weights.append(conf)
                                        # Add expression value to a temporary position in X_orig
                                        if virtual_protein not in protein_to_idx:
                                            protein_to_idx[virtual_protein] = -1  # Mark as temporary
                                            X_orig = np.hstack((X_orig, rna_expr.reshape(-1, 1)))
                                            protein_to_idx[virtual_protein] = X_orig.shape[1] - 1
                                        rna_interactors += 1

                        print(f"  Found {rna_interactors} interacting genes in RNA data")

                    # 3. If still no interacting partners found, try secondary interactions
                    if len(interacting_proteins) == 0:
                        print("  Trying to find secondary interacting proteins...")
                        secondary_interactors = 0
                        # For each interacting protein, sort to ensure consistency
                        for p2 in sorted(ppi_network[string_id].keys()):
                            conf1 = ppi_network[string_id][p2]
                            if conf1 >= confidence_threshold:
                                # Find its interacting proteins
                                if p2 in ppi_network:
                                    # Sort secondary interactions to ensure consistency
                                    for p3 in sorted(ppi_network[p2].keys()):
                                        conf2 = ppi_network[p2][p3]
                                        if conf2 >= confidence_threshold and p3 in string_to_adt:
                                            # Use combined confidence
                                            combined_conf = conf1 * conf2
                                            if combined_conf >= confidence_threshold:
                                                interacting_proteins.append(string_to_adt[p3])
                                                interaction_weights.append(combined_conf)
                                                secondary_interactors += 1

                        print(f"  Found {secondary_interactors} secondary interacting proteins")

            if not interacting_proteins:
                print(f"  No interacting proteins found, skipping")
                continue

            print(f"  Found {len(interacting_proteins)} interacting proteins, confidence threshold: {confidence_threshold}")

            # Calculate weighted average expression and confidence
            weighted_expr = np.zeros(X_orig.shape[0])
            confidence_scores = np.zeros(X_orig.shape[0])  # Used to store confidence for each position
            total_weight = 0

            for p2, weight in zip(interacting_proteins, interaction_weights):
                p2_idx = protein_to_idx[p2]
                weighted_expr += weight * X_orig[:, p2_idx]
                confidence_scores += weight  # Accumulate weights as confidence
                total_weight += weight

            if total_weight > 0:
                predicted_expr = weighted_expr / total_weight
                confidence_scores = confidence_scores / total_weight  # Normalize confidence

                # Only impute positions that are zero in original data and have confidence above threshold
                original_zero_mask = X_orig[:, protein_idx] == 0
                high_confidence_mask = confidence_scores >= min_confidence_for_imputation

                # Find all potentially fillable positions (zero and predicted value > 0)
                potential_fill_mask = original_zero_mask & high_confidence_mask & (predicted_expr > 0)
                potential_fill_positions = np.where(potential_fill_mask)[0]
                potential_fill_values = predicted_expr[potential_fill_positions]

                # Only select top N% positions by predicted value
                if len(potential_fill_positions) > 0:
                    # Calculate number of positions to select
                    select_count = max(1, int(len(potential_fill_positions) * top_percent_to_impute / 100))

                    # Sort by predicted value, select highest
                    sorted_indices = np.argsort(potential_fill_values)[::-1]  # Descending order
                    top_indices = sorted_indices[:select_count]
                    top_positions = potential_fill_positions[top_indices]

                    # Create final imputation mask
                    fillable_mask = np.zeros_like(original_zero_mask, dtype=bool)
                    fillable_mask[top_positions] = True
                else:
                    fillable_mask = potential_fill_mask

                zero_count = np.sum(original_zero_mask)
                high_conf_count = np.sum(original_zero_mask & high_confidence_mask)
                print(f"  Original zero count: {zero_count}")
                print(f"  High confidence zero count: {high_conf_count}")

                # Check predicted values
                pred_nonzero = np.sum(predicted_expr > 0)
                print(f"  Predicted non-zero count: {pred_nonzero}")

                # Calculate fillable value count
                potential_changes = np.sum(fillable_mask)
                print(f"  Fillable value count: {potential_changes}")
                print(f"  Selected top {top_percent_to_impute}% positions by predicted value, total {potential_changes}")

                # Calculate imputation rate and apply upper limit
                if zero_count > 0:
                    # Calculate true imputation rate = imputed count / total sample count
                    impute_rate = potential_changes / total_samples
                    print(f"  Potential imputation rate (relative to total samples): {impute_rate:.2%}")

                    # If imputation rate exceeds limit, select positions with higher predicted values
                    if impute_rate > max_impute_rate and potential_changes > 0:
                        print(f"  Imputation rate exceeds limit {max_impute_rate*100}%, will select positions with higher predicted values")

                        # Find all fillable positions
                        fillable_positions = np.where(fillable_mask)[0]

                        # Get predicted values at these positions
                        fillable_values = predicted_expr[fillable_positions]

                        # Calculate maximum allowed imputation count
                        max_fill_count = int(total_samples * max_impute_rate)

                        # Sort by predicted value, select positions with highest values
                        sorted_indices = np.argsort(fillable_values)[::-1]  # Descending order
                        selected_indices = sorted_indices[:min(max_fill_count, len(fillable_positions))]
                        selected_positions = fillable_positions[selected_indices]

                        # Create new mask, only impute at selected positions
                        new_mask = np.zeros_like(original_zero_mask, dtype=bool)
                        new_mask[selected_positions] = True

                        # Apply new mask
                        changes = np.sum(new_mask)
                        X_orig[new_mask, protein_idx] = predicted_expr[new_mask]

                        # Record actual imputation rate
                        actual_impute_rate = changes / total_samples
                        protein_impute_rates[protein_name] = actual_impute_rate * 100

                        print(f"  Actually imputed {changes} values, imputation rate (relative to total samples): {actual_impute_rate:.2%}")
                        print(f"  Average imputed value: {np.mean(predicted_expr[new_mask]):.4f}")
                    else:
                        # Imputation rate within allowed range, impute directly
                        changes = potential_changes
                        X_orig[original_zero_mask & (predicted_expr > 0), protein_idx] = predicted_expr[original_zero_mask & (predicted_expr > 0)]

                        # Record actual imputation rate
                        actual_impute_rate = changes / total_samples
                        protein_impute_rates[protein_name] = actual_impute_rate * 100

                        print(f"  Imputed {changes} values, imputation rate (relative to total samples): {actual_impute_rate:.2%}")

                    if changes > 0:
                        filled_proteins += 1
                        filled_values += changes
                        filled_protein_names.append(protein_name)
                else:
                    print(f"  No zeros to impute")
                    protein_impute_rates[protein_name] = 0.0
            else:
                print(f"  Cannot calculate predicted value, skipping")
                protein_impute_rates[protein_name] = 0.0

        # Update original data matrix
        adata.X = scipy.sparse.csr_matrix(X_orig)

        # Calculate imputation effect
        print(f"\nImputation statistics:")
        print(f"- Successfully imputed {filled_proteins} proteins")
        print(f"- Total {filled_values} values imputed")
        print(f"- Non-zero element ratio after imputation: {np.count_nonzero(X_orig) / X_orig.size:.2%}")

        # Output successfully imputed protein names and imputation rates
        print(f"\nSuccessfully imputed protein names and imputation rates:")

        # Sort by imputation rate
        sorted_proteins_by_rate = sorted(protein_impute_rates.items(), key=lambda x: x[1], reverse=True)

        # Output top 10 proteins by imputation rate
        print(f"\nTop 10 proteins by imputation rate:")
        for i, (protein, rate) in enumerate(sorted_proteins_by_rate[:10]):
            source = "unknown"
            string_ids = ""

            # Get mapping source and STRING ID
            if 'mapping_source' in adata.var.columns and 'string_ids' in adata.var.columns:
                idx = np.where(adata.var_names == protein)[0]
                if len(idx) > 0:
                    source = adata.var['mapping_source'].iloc[idx[0]]
                    string_ids = adata.var['string_ids'].iloc[idx[0]]

            print(f"- {protein}: imputation rate {rate:.2f}%, mapping source: {source}, STRING ID: {string_ids}")

        # Add imputation rate info to adata.var
        impute_rates = np.zeros(adata.n_vars)
        for protein, rate in protein_impute_rates.items():
            if protein in adata.var_names:
                idx = np.where(adata.var_names == protein)[0][0]
                impute_rates[idx] = rate

        adata.var['impute_rate'] = impute_rates

        return adata


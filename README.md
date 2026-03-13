# Prior knowledge-enhanced deep learning enables comprehensive spatial multi-omics integration

![Framework](./figures/figure1.jpg)

# Installation & Dependencies
You'll need to install the following packages in order to run the codes.


| Package         | Version    |
|-----------------|------------|
| python          | 3.11.3     |
| pytorch         | 2.1.0      |
| cuda            | 12.1       |
| torchvision     | 0.16.0     |
| torchaudio      | 2.1.0      |
| torch-scatter   | 2.1.2      |
| torch-sparse    | 0.6.18     |
| torch_geometric | 2.5.2      |
| numpy           | 1.26.4     |
| pandas          | 2.2.3      |
| scanpy          | 1.9.3      |
| anndata         | 0.9.1      |
| scipy           | 1.14.1     |
| scikit-learn    | 1.7.0      |
| seaborn         | 0.13.2     |
| tqdm            | 4.66.5     |
| rpy2            | 3.4.1      |
| R               | 4.4.1      |
| mclust (R)      | 6.1.2      |

# Tutorial
For the step-by-step tutorial, please refer to:https://github.com/Bio-wust/DeepSpak/tree/main/tutorial

# Dataset

This project supports biology-guided imputation for three types of omics data. The data sources are as follows:

## RNA Imputation
- **Source**: [TRRUST](https://www.grtoolszen.com/trrust/) (Transcriptional Regulatory Relationships Unraveled by Sentence-based Text mining)
- **Usage**: Predict and impute expression values of low-expression genes using the Gene Regulatory Network (GRN)

## ADT (Protein) Imputation
- **Source**: [STRING](https://string-db.org/) (Search Tool for the Retrieval of Interacting Genes/Proteins)
- **Usage**: Predict and impute protein expression data using the Protein-Protein Interaction (PPI) network

## ATAC Imputation
- **Source**: [GENCODE](https://www.gencodegenes.org/) (Gene and Variation Annotation)
- **Usage**: Predict chromatin accessibility using Peak-TF (Transcription Factor binding site) mapping relationships based on TF expression levels

## Data File Acquisition and Usage

### peak_tf_db (Peak-TF mapping file)
- **Description**: Mapping file between ATAC peaks and transcription factors
- **Format**: BED/CSV/JSON format
- **Acquisition**: Use the provided pipeline script `data_process/peak_tf_pipeline.py`
  1. Download gene annotation from [GENCODE](https://www.gencodegenes.org/)
  2. Run the 3-step pipeline:
```bash
# Run full pipeline (all 3 steps)
python data_process/peak_tf_pipeline.py --mode all \
    --gtf_input path/to/gencode.annotation.gff3 \
    --atac_input path/to/adata_atac.h5ad \
    --mapping_output path/to/peaks_to_tf.bed
```
- **Usage**:
```python
data_loader = SpatialOmicsDataLoader(
    data_dir='your_data_dir/',
    peak_tf_db_path='path/to/peaks_to_tf.bed'  # BED, CSV, or JSON format
)
adata_atac = data_loader.impute_atac_with_peak_tf(adata_rna, adata_atac)
```

### string_protein_info_file (STRING protein info)
- **Description**: Protein information file from STRING database for protein ID mapping
- **Download**: [STRING Database](https://string-db.org/cgi/download.pl)
  - Select your species (e.g., Homo sapiens, Mus musculus)
  - Download "protein.info.vXX.txt.gz" file
- **Usage**:
```python
data_loader = SpatialOmicsDataLoader(
    data_dir='your_data_dir/',
    string_protein_info_file='path/to/protein.info.v12.0.txt'
)
```

### ppi_file (PPI network file)
- **Description**: Protein-protein interaction network from STRING database
- **Download**: [STRING Database](https://string-db.org/cgi/download.pl)
  - Select your species
  - Download "protein.links.vXX.txt.gz" file (full network)
  - Or "protein.links.full.vXX.txt.gz" for detailed scores
- **Format**: TSV with columns: protein1, protein2, combined_score
- **Usage**:
```python
data_loader = SpatialOmicsDataLoader(
    data_dir='your_data_dir/',
    string_protein_info_file='path/to/protein.info.v12.0.txt',
    ppi_file_path='path/to/protein.links.v12.0.txt'
)
adata_protein = data_loader.impute_protein_with_ppi(adata_rna, adata_protein)
```



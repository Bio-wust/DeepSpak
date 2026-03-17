import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class ProteinIDMapper:
    """
    Protein ID mapper for mapping protein names in ADT data to STRING database protein IDs
    """

    def __init__(self, manual_mapping=None, string_protein_info_file=None):
        """
        Initialize the protein ID mapper

        Parameters:
            manual_mapping: Manual mapping dictionary, format: {protein_name: [gene_id, [protein_id_list]]}
            string_protein_info_file: STRING database protein info file path, e.g., 9606.protein.info.v12.0.txt
        """
        # Manual mapping is now empty, fully relying on accurate STRING database mapping
        self.manual_mapping = manual_mapping or {}
        # self.manual_mapping = manual_mapping or {
        #     # Protein name: [gene_id, protein_id_list]
        #     'CD163': ['ENSG00000177575', ['ENSP00000359722', 'ENSP00000359723']],
        #     'CR2': ['ENSG00000117322', ['ENSP00000367890', 'ENSP00000484480']],
        #     'PCNA': ['ENSG00000132646', ['ENSP00000368438', 'ENSP00000379472']],
        #     'VIM': ['ENSG00000026025', ['ENSP00000224237']],
        #     'KRT5': ['ENSG00000186081', ['ENSP00000252242']],
        #     'CD68': ['ENSG00000129226', ['ENSP00000358022']],
        #     'CEACAM8': ['ENSG00000124469', ['ENSP00000244405']],
        #     'PTPRC': ['ENSG00000081237', ['ENSP00000352288', 'ENSP00000352304']],
        #     'HLA-DRA': ['ENSG00000204287', ['ENSP00000364114']],
        #     'PAX5': ['ENSG00000196092', ['ENSP00000350844']],
        #     'SDC1': ['ENSG00000115884', ['ENSP00000254351']],
        #     'CD8A': ['ENSG00000153563', ['ENSP00000283635']],
        #     'BCL2': ['ENSG00000171791', ['ENSP00000329623']],
        #     'CD19': ['ENSG00000177455', ['ENSP00000352455']],
        #     'PDCD1': ['ENSG00000188389', ['ENSP00000335062']],
        #     'ACTA2': ['ENSG00000107796', ['ENSP00000224784']],
        #     'FCGR3A': ['ENSG00000203747', ['ENSP00000368664']],
        #     'ITGAX': ['ENSG00000140678', ['ENSP00000380227']],
        #     'CXCR5': ['ENSG00000160683', ['ENSP00000292174']],
        #     'EPCAM': ['ENSG00000119888', ['ENSP00000263735']],
        #     'MS4A1': ['ENSG00000156738', ['ENSP00000358566']],  # CD20
        #     'CD3E': ['ENSG00000198851', ['ENSP00000354566']],
        #     'CD14': ['ENSG00000170458', ['ENSP00000364190']],
        #     'CD40': ['ENSG00000101017', ['ENSP00000361359']],
        #     'PECAM1': ['ENSG00000261371', ['ENSP00000457421']],  # CD31
        #     'CD4': ['ENSG00000010610', ['ENSP00000011653']],
        #     'ITGAM': ['ENSG00000169896', ['ENSP00000305700']],  # CD11b
        #     'CD27': ['ENSG00000139193', ['ENSP00000266557']],
        #     'CCR7': ['ENSG00000126353', ['ENSP00000246657']],
        #     'CD274': ['ENSG00000120217', ['ENSP00000384410']],   # PD-L1
        #     # Additional protein mappings
        #     'CD45RO': ['ENSG00000081237', ['ENSP00000352288']],  # PTPRC variant
        #     'THY1': ['ENSG00000154096', ['ENSP00000367273']],    # CD90
        #     'FOXP3': ['ENSG00000049768', ['ENSP00000365380']],
        #     'ICOS': ['ENSG00000163600', ['ENSP00000370016']]
        # }


        # Load protein info from STRING database
        self.string_protein_info = {}
        self.preferred_name_to_string_id = {}

        if string_protein_info_file and os.path.exists(string_protein_info_file):
            self.load_string_protein_info(string_protein_info_file)

        # Create protein name to STRING ID mapping (now mainly relies on STRING database)
        self.protein_to_string_ids = {}
        for protein, mapping_data in self.manual_mapping.items():
            if isinstance(mapping_data, list) and len(mapping_data) >= 2:
                # Compatible with original format: [gene_id, [protein_id_list]]
                gene_id, protein_ids = mapping_data[0], mapping_data[1]
                string_ids = protein_ids if isinstance(protein_ids, list) else [protein_ids]
                self.protein_to_string_ids[protein] = string_ids

        # Create STRING ID to protein name mapping
        self.string_to_protein = {}
        for protein, string_ids in self.protein_to_string_ids.items():
            for string_id in string_ids:
                self.string_to_protein[string_id] = protein
    
    def load_string_protein_info(self, file_path):
        """
        Load protein info from STRING database file
        Only keeps string_protein_id to preferred_name mapping

        Standard format: #string_protein_id	preferred_name	protein_size	annotation

        Parameters:
            file_path: STRING database protein info file path
        """
        print(f"Loading protein info from STRING database: {file_path}")

        # Detect species type
        species_code = "unknown"
        if "9606" in file_path:
            species_code = "human"
        elif "10090" in file_path:
            species_code = "mouse"
        print(f"Detected species: {species_code}")

        try:
            # STRING file has fixed header format: #string_protein_id	preferred_name	protein_size	annotation
            # First read header line to get column names, then read data
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()

            if header_line.startswith('#'):
                # Extract column names (remove # symbol)
                column_names = header_line[1:].strip().split('\t')
                print(f"Detected header line, column names: {column_names}")

                # Only read first two columns: string_protein_id and preferred_name
                df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None,
                               names=column_names, usecols=[0, 1])
            else:
                # No header line, use default column names, only read first two columns
                print("No header line detected, using default column names")
                df = pd.read_csv(file_path, sep='\t', header=None,
                               names=['string_protein_id', 'preferred_name'], usecols=[0, 1])

            print(f"Successfully loaded STRING protein info: {len(df)} rows")
            print(f"Actual column names: {list(df.columns)}")

        except Exception as e:
            print(f"Failed to read file: {e}")
            import traceback
            traceback.print_exc()
            return

        # Verify required columns exist
        if 'string_protein_id' in df.columns and 'preferred_name' in df.columns:
            # Create STRING ID to protein info mapping
            for _, row in df.iterrows():
                string_id = row['string_protein_id']
                preferred_name = row['preferred_name']

                # Normalize protein ID format (remove species prefix if exists)
                clean_string_id = string_id
                if '.' in string_id and (string_id.startswith('9606.') or string_id.startswith('10090.')):
                    clean_string_id = string_id.split('.', 1)[1]

                # Simplified storage: only save preferred_name (use cleaned ID as primary key)
                self.string_protein_info[clean_string_id] = preferred_name

                # If original ID differs from cleaned ID, also store original ID mapping
                if string_id != clean_string_id:
                    self.string_protein_info[string_id] = preferred_name

                # Create preferred name to STRING ID mapping (use cleaned ID)
                if preferred_name not in self.preferred_name_to_string_id:
                    self.preferred_name_to_string_id[preferred_name] = []
                if clean_string_id not in self.preferred_name_to_string_id[preferred_name]:
                    self.preferred_name_to_string_id[preferred_name].append(clean_string_id)

            print(f"Processed {len(self.string_protein_info)} protein info entries")
            print(f"Created {len(self.preferred_name_to_string_id)} preferred name to STRING ID mappings")

            # Print some examples
            sample_ids = list(self.string_protein_info.keys())[:5]
            print("\nSample protein info:")
            for string_id in sample_ids:
                preferred_name = self.string_protein_info[string_id]
                print(f"  {string_id}: {preferred_name}")
        else:
            print(f"Error: Incorrect file format, column names: {list(df.columns)}")
    
    def extract_gene_symbols_from_annotation(self):
        """
        Build gene symbol mapping from preferred names (simplified version, no longer uses annotations)
        """
        print("Building gene symbol mapping from preferred names...")

        gene_symbol_to_string_ids = {}

        for string_id, preferred_name in self.string_protein_info.items():
            # Use preferred name directly as gene symbol
            if preferred_name:
                if preferred_name not in gene_symbol_to_string_ids:
                    gene_symbol_to_string_ids[preferred_name] = []
                if string_id not in gene_symbol_to_string_ids[preferred_name]:
                    gene_symbol_to_string_ids[preferred_name].append(string_id)

        print(f"Built {len(gene_symbol_to_string_ids)} gene symbol mappings")

        # Update mapping
        self.gene_symbol_to_string_ids = gene_symbol_to_string_ids

        # Print some examples
        sample_symbols = list(gene_symbol_to_string_ids.keys())[:10]
        print("\nSample gene symbol mappings:")
        for symbol in sample_symbols:
            string_ids = gene_symbol_to_string_ids[symbol][:3]  # Show at most 3
            print(f"  {symbol}: {string_ids}")

        return gene_symbol_to_string_ids
    
    def map_protein_to_string_ids(self, protein_name):
        """
        Map protein name to STRING ID

        Parameters:
            protein_name: Protein name

        Returns:
            List of STRING IDs, Returns empty list if not found
        """
        # 1. Priority check: preferred name mapping in STRING database (most accurate)
        if protein_name in self.preferred_name_to_string_id:
            return self.preferred_name_to_string_id[protein_name]

        # 2. Check gene symbol mapping
        if hasattr(self, 'gene_symbol_to_string_ids') and protein_name in self.gene_symbol_to_string_ids:
            return self.gene_symbol_to_string_ids[protein_name]

        # 3. Fallback: Check manual mapping (may be outdated)
        if protein_name in self.protein_to_string_ids:
            return self.protein_to_string_ids[protein_name]

        # 4. Fuzzy matching - Check if preferred name contains protein name
        matches = []
        for name, string_ids in self.preferred_name_to_string_id.items():
            if protein_name.lower() in name.lower() or name.lower() in protein_name.lower():
                matches.extend(string_ids)

        if matches:
            return matches

        return []
    
    def map_string_id_to_protein(self, string_id):
        """
        Map STRING ID to protein name

        Parameters:
            string_id: STRING ID

        Returns:
            Protein name. Returns None if not found
        """
        # 1. Check manual mapping
        if string_id in self.string_to_protein:
            return self.string_to_protein[string_id]

        # 2. Check STRING protein info (now directly stores preferred_name)
        if string_id in self.string_protein_info:
            return self.string_protein_info[string_id]

        return None
    
    def find_protein_in_ppi_network(self, protein_name, ppi_network):
        """
        Find protein in PPI network

        Parameters:
            protein_name: Protein name
            ppi_network: PPI network, format: {string_id: {string_id: score}}

        Returns:
            (found_or_not, corresponding_STRING_ID_list)
        """
        # 1. Use mapping method to find STRING ID
        string_ids = self.map_protein_to_string_ids(protein_name)
        if string_ids:
            # Check if these IDs are in PPI network
            found_ids = [sid for sid in string_ids if sid in ppi_network]
            if found_ids:
                return True, found_ids

        # 2. Try direct matching
        if protein_name in ppi_network:
            return True, [protein_name]

        # 3. Fuzzy matching - Check if STRING ID contains protein name
        for string_id in ppi_network.keys():
            # Remove species prefix
            if '.' in string_id:
                base_id = string_id.split('.')[1]
            else:
                base_id = string_id

            # Check if it contains protein name
            if protein_name.lower() in base_id.lower():
                return True, [string_id]

        # 4. More lenient fuzzy matching - Use prefix
        if len(protein_name) > 3:
            prefix = protein_name[:3].lower()
            for string_id in ppi_network.keys():
                # Remove species prefix
                if '.' in string_id:
                    base_id = string_id.split('.')[1]
                else:
                    base_id = string_id

                # Check if it contains protein name prefix
                if prefix in base_id.lower():
                    return True, [string_id]

        return False, []
    
    def create_mapping_for_adt_data(self, adt_proteins, ppi_network=None, gene_to_ensembl=None):
        """
        Create mapping for proteins in ADT data

        Parameters:
            adt_proteins: List of protein names in ADT data
            ppi_network: Optional, PPI network, format: {string_id: {string_id: score}}
            gene_to_ensembl: Optional, gene symbol to Ensembl ID mapping

        Returns:
            Mapping dictionary, format: {protein_name: {'source': source, 'gene_id': gene_id, 'string_ids': STRING_ID_list}}
        """
        # If STRING protein info exists but gene symbols not extracted, extract first
        if self.string_protein_info and not hasattr(self, 'gene_symbol_to_string_ids'):
            self.extract_gene_symbols_from_annotation()

        mapping = {}

        for protein in adt_proteins:
            # 1. Use manual mapping
            if protein in self.manual_mapping:
                mapping_data = self.manual_mapping[protein]
                if isinstance(mapping_data, list) and len(mapping_data) >= 2:
                    gene_id, protein_ids = mapping_data[0], mapping_data[1]
                    string_ids = protein_ids if isinstance(protein_ids, list) else [protein_ids]
                    mapping[protein] = {
                        'source': 'manual_mapping',
                        'gene_id': gene_id,
                        'string_ids': string_ids
                    }
                    continue

            # 2. Use STRING database mapping
            string_ids = self.map_protein_to_string_ids(protein)
            if string_ids:
                # Try to find gene ID
                gene_id = None
                if gene_to_ensembl and protein in gene_to_ensembl:
                    gene_id = gene_to_ensembl[protein]

                mapping[protein] = {
                    'source': 'string_database',
                    'gene_id': gene_id,
                    'string_ids': string_ids
                }
                continue

            # 3. Use PPI network to find
            if ppi_network:
                found, string_ids = self.find_protein_in_ppi_network(protein, ppi_network)
                if found:
                    mapping[protein] = {
                        'source': 'ppi_network',
                        'gene_id': None,  # PPI network usually doesn't have gene ID
                        'string_ids': string_ids
                    }
                    continue

            # 4. Use gene symbol to Ensembl ID mapping
            if gene_to_ensembl and protein in gene_to_ensembl:
                gene_id = gene_to_ensembl[protein]
                # Guess protein ID from gene ID
                protein_id = gene_id.replace('ENSG', 'ENSP')
                string_id = protein_id
                mapping[protein] = {
                    'source': 'gene_mapping',
                    'gene_id': gene_id,
                    'string_ids': [string_id]
                }
                continue

            # 5. Mapping not found
            mapping[protein] = {
                'source': 'not_found',
                'gene_id': None,
                'string_ids': []
            }

        return mapping
    
    def print_mapping_stats(self, mapping):
        """
        Print mapping statistics

        Parameters:
            mapping: Mapping dictionary
        """
        sources = {}
        for protein, info in mapping.items():
            source = info['source']
            if source not in sources:
                sources[source] = []
            sources[source].append(protein)

        print("\nMapping statistics:")
        for source, proteins in sources.items():
            print(f"{source}: {len(proteins)} proteins")
            if len(proteins) > 0:
                print(f"  Examples: {proteins[:min(3, len(proteins))]}")

        # Calculate total mapping rate
        total = len(mapping)
        mapped = total - len(sources.get('not_found', []))
        print(f"\nTotal mapping rate: {mapped}/{total} ({mapped/total:.2%})")

def main():
    """Main function for testing the protein ID mapper"""
    # Set STRING database file path
    string_protein_info_file = 'data/Data_SpatialGlue/Dataset1_Mouse_Spleen1/10090.protein.info.v12.0.txt'

    # Create mapper
    mapper = ProteinIDMapper(string_protein_info_file=string_protein_info_file)

    # Test ADT protein list
    adt_proteins = [
        'CD163', 'CR2', 'PCNA', 'VIM', 'KRT5', 'CD68', 'CEACAM8', 'PTPRC', 'HLA-DRA', 'PAX5',
        'SDC1', 'CD8A', 'BCL2', 'CD19', 'PDCD1', 'ACTA2', 'FCGR3A', 'ITGAX', 'CXCR5', 'EPCAM',
        'MS4A1', 'CD3E', 'CD14', 'CD40', 'PECAM1', 'CD4', 'ITGAM', 'CD27', 'CCR7', 'CD274'
    ]

    # Create mapping
    mapping = mapper.create_mapping_for_adt_data(adt_proteins)

    # Print mapping statistics
    mapper.print_mapping_stats(mapping)

    # Print detailed mapping
    print("\nDetailed mapping:")
    for protein, info in mapping.items():
        if info['source'] != 'not_found':
            print(f"{protein}: {info['string_ids']}")

if __name__ == "__main__":
    main() 
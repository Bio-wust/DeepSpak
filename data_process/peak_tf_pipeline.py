"""
Peak-TF Mapping Analysis Pipeline
Integrates three steps:
1. Create gene info BED file from GTF/GFF annotation file
2. Extract peak information from ATAC-seq data and save as BED format
3. Create peak-TF mapping relationships

Usage:
    python peak_tf_pipeline.py --mode all                        # Run full pipeline
    python peak_tf_pipeline.py --mode create_gene_info           # Run step 1 only
    python peak_tf_pipeline.py --mode extract_peaks              # Run step 2 only
    python peak_tf_pipeline.py --mode create_mapping             # Run step 3 only
"""

import argparse
import gzip
import os
import re
import sys
import json
import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm


# =============================================================================
# Step 1: Create gene info BED file from GTF/GFF file
# =============================================================================

def parse_gtf_attributes(attr_string):
    """Parse attribute string in GTF/GFF format"""
    attributes = {}

    # Detect file format
    if "=" in attr_string:  # GFF3 format: key=value;
        attrs = attr_string.split(';')
        for attr in attrs:
            attr = attr.strip()
            if not attr:
                continue
            if '=' in attr:
                key, value = attr.split('=', 1)
                attributes[key] = value
    else:  # GTF format: key "value";
        pattern = re.compile(r'(\w+)\s+"([^"]+)";')
        for match in pattern.finditer(attr_string):
            key, value = match.groups()
            attributes[key] = value

    return attributes


def load_annotation(gtf_file, feature_type, gene_id_attr, gene_name_attr):
    """Load annotation file and extract records of specified feature type"""
    print(f"Loading annotation file: {gtf_file}")
    gene_records = []

    is_gzipped = gtf_file.endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'

    try:
        with open_func(gtf_file, mode) as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 9:
                    print(f"Warning: Line {line_num} has incorrect format, skipping")
                    continue

                if fields[2] == feature_type:
                    chrom = fields[0]
                    start = int(fields[3]) - 1  # Convert to 0-based
                    end = int(fields[4])
                    strand = fields[6]

                    attributes = parse_gtf_attributes(fields[8])
                    gene_id = attributes.get(gene_id_attr, 'unknown_id')
                    gene_name = attributes.get(gene_name_attr, gene_id)

                    gene_records.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'name': gene_name,
                        'id': gene_id,
                        'strand': strand
                    })

        print(f"Loaded {len(gene_records)} {feature_type} records")
        return gene_records

    except Exception as e:
        print(f"Error loading annotation file: {e}")
        sys.exit(1)


def create_bed_records(gene_records, include_promoters, promoter_upstream, promoter_downstream):
    """Create BED records, optionally including promoter regions"""
    bed_records = []

    for gene in gene_records:
        bed_records.append([
            gene['chrom'],
            str(gene['start']),
            str(gene['end']),
            gene['name'],
            '1000',
            gene['strand']
        ])

        if include_promoters:
            if gene['strand'] == '+':
                prom_start = max(0, gene['start'] - promoter_upstream)
                prom_end = gene['start'] + promoter_downstream
                promoter_name = f"{gene['name']}_promoter"

                bed_records.append([
                    gene['chrom'],
                    str(prom_start),
                    str(prom_end),
                    promoter_name,
                    '800',
                    gene['strand']
                ])
            else:
                prom_start = max(0, gene['end'] - promoter_downstream)
                prom_end = gene['end'] + promoter_upstream
                promoter_name = f"{gene['name']}_promoter"

                bed_records.append([
                    gene['chrom'],
                    str(prom_start),
                    str(prom_end),
                    promoter_name,
                    '800',
                    gene['strand']
                ])

    return bed_records


def save_bed_file_from_records(bed_records, output_file):
    """Save BED file"""
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w') as f:
            for record in bed_records:
                f.write('\t'.join(record) + '\n')

        print(f"Successfully saved BED file to: {output_file}")
        print(f"Created {len(bed_records)} records")
        return True
    except Exception as e:
        print(f"Error saving BED file: {e}")
        return False


def step1_create_gene_info(args):
    """Step 1: Create gene info BED file"""
    print("\n" + "=" * 60)
    print("Step 1: Create gene info BED file")
    print("=" * 60)

    gene_records = load_annotation(
        args.gtf_input,
        args.feature_type,
        args.gene_id_attr,
        args.gene_name_attr
    )

    bed_records = create_bed_records(
        gene_records,
        args.include_promoters,
        args.promoter_upstream,
        args.promoter_downstream
    )

    success = save_bed_file_from_records(bed_records, args.gene_info_output)

    if success:
        promoter_info = "including promoter regions" if args.include_promoters else "without promoter regions"
        print(f"Step 1 completed: {promoter_info}")

    return success


# =============================================================================
# Step 2: Extract peak information from ATAC-seq data
# =============================================================================

def format_peak_name(chrom, start, end, format_type='hyphen'):
    """Format peak name"""
    if format_type == 'colon':
        return f"{chrom}:{start}-{end}"
    elif format_type == 'underscore':
        return f"{chrom}_{start}_{end}"
    else:
        return f"{chrom}-{start}-{end}"


def extract_chrom_pos(peak_name):
    """Extract chromosome, start and end positions from peak name"""
    pattern1 = r'(chr[\dXY]+):(\d+)-(\d+)'
    pattern2 = r'(chr[\dXY]+)_(\d+)_(\d+)'
    pattern3 = r'(chr[\dXY]+)\.(\d+)\.(\d+)'
    pattern4 = r'(chr[\dXY]+)-(\d+)-(\d+)'

    for pattern in [pattern1, pattern2, pattern3, pattern4]:
        match = re.match(pattern, peak_name)
        if match:
            chrom, start, end = match.groups()
            return chrom, int(start), int(end)

    parts = re.split(r'[:_\.-]', peak_name)
    if len(parts) >= 3 and parts[0].startswith('chr'):
        try:
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            return chrom, start, end
        except ValueError:
            pass

    return None


def load_atac_data(file_path):
    """Load ATAC-seq data"""
    print(f"Loading ATAC-seq data: {file_path}")
    try:
        adata = sc.read_h5ad(file_path)
        print(f"Successfully loaded data: {adata.shape[0]} cells, {adata.shape[1]} peaks")
        return adata
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)


def extract_peaks(adata, min_expression=0.0, name_format='hyphen'):
    """Extract all valid peak information and sort by expression"""
    print(f"Extracting peak information" + (f", expression threshold: {min_expression}" if min_expression > 0 else ""))

    peak_names = adata.var_names.tolist()

    if hasattr(adata.X, 'toarray'):
        scores = np.array(adata.X.mean(axis=0)).flatten()
    else:
        scores = np.array(adata.X.mean(axis=0))

    peaks_df = pd.DataFrame({
        'peak_name': peak_names,
        'expression': scores
    })

    if min_expression > 0:
        peaks_df = peaks_df[peaks_df['expression'] > min_expression]

    peaks_df = peaks_df.sort_values('expression', ascending=False)

    print("Parsing peak names and extracting chromosome coordinates...")
    valid_peaks = []
    invalid_peaks = 0

    for _, row in tqdm(peaks_df.iterrows(), total=len(peaks_df)):
        peak_name = row['peak_name']
        coords = extract_chrom_pos(peak_name)

        if coords is not None:
            chrom, start, end = coords
            if chrom.startswith('chr') and (chrom[3:].isdigit() or chrom[3:] in ['X', 'Y']):
                formatted_name = format_peak_name(chrom, start, end, name_format)
                valid_peaks.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'name': formatted_name,
                    'score': row['expression'],
                    'strand': '.'
                })
        else:
            invalid_peaks += 1

    result_df = pd.DataFrame(valid_peaks)
    result_df = result_df.sort_values(['chrom', 'start'])

    print(f"Extracted {len(result_df)} valid peaks" + (f", skipped {invalid_peaks} invalid peaks" if invalid_peaks > 0 else ""))
    return result_df


def save_peaks_bed_file(peaks_df, output_file):
    """Save peak information to BED file"""
    print(f"Saving BED file: {output_file}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    peaks_df[['chrom', 'start', 'end', 'name', 'score', 'strand']].to_csv(
        output_file, sep='\t', header=False, index=False
    )

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Successfully saved {len(peaks_df)} peaks to BED file, file size: {file_size_mb:.2f} MB")

    return file_size_mb


def step2_extract_peaks(args):
    """Step 2: Extract peak information from ATAC-seq data"""
    print("\n" + "=" * 60)
    print("Step 2: Extract peak information from ATAC-seq data")
    print("=" * 60)

    adata = load_atac_data(args.atac_input)
    peaks_df = extract_peaks(adata, args.min_expression, args.name_format)
    save_peaks_bed_file(peaks_df, args.peaks_output)

    print("Step 2 completed!")
    return True


# =============================================================================
# Step 3: Create peak-TF mapping relationships
# =============================================================================

def load_tf_list(tf_list_file):
    """Load transcription factor list"""
    try:
        if tf_list_file is None:
            return None

        tf_set = set()
        with open(tf_list_file, 'r') as f:
            for line in f:
                tf = line.strip()
                if tf:
                    tf_set.add(tf)

        print(f"Loaded {len(tf_set)} transcription factors")
        return tf_set

    except Exception as e:
        print(f"Error loading transcription factor list: {e}")
        return None


def load_bed_file(bed_file, names=None):
    """Load BED file into DataFrame"""
    try:
        if names is None:
            names = ['chrom', 'start', 'end', 'name', 'score', 'strand']

        df = pd.read_csv(bed_file, sep='\t', header=None, comment='#')

        if len(df.columns) >= 6:
            df.columns = names + [f'col{i+7}' for i in range(len(df.columns)-6)]
        elif len(df.columns) == 5:
            df.columns = names[:5]
            df['strand'] = '.'
        elif len(df.columns) == 4:
            df.columns = names[:4]
            df['score'] = 0
            df['strand'] = '.'
        elif len(df.columns) == 3:
            df.columns = names[:3]
            df['name'] = df.apply(lambda x: f"{x['chrom']}-{x['start']}-{x['end']}", axis=1)
            df['score'] = 0
            df['strand'] = '.'
        else:
            raise ValueError(f"Incorrect BED file format, column count: {len(df.columns)}")

        print(f"Successfully loaded BED file: {bed_file}, containing {len(df)} records")
        return df

    except Exception as e:
        print(f"Error loading BED file: {e}")
        sys.exit(1)


def filter_gene_info_for_tfs(gene_df, tf_set):
    """Filter gene info to keep only transcription factors"""
    if tf_set is None:
        print("No transcription factor list specified, using all genes")
        return gene_df

    try:
        exact_matches = sum(1 for name in gene_df['name'] if name in tf_set)
        print(f"Number of exactly matched transcription factors: {exact_matches}")

        if exact_matches < 10:
            print("Few exact matches, attempting partial matching...")
            matched_genes = []

            for _, row in gene_df.iterrows():
                gene_name = row['name']

                if gene_name in tf_set:
                    matched_genes.append(row)
                    continue

                matched = False
                for tf in tf_set:
                    if tf in gene_name or gene_name in tf or tf.lower() == gene_name.lower():
                        matched = True
                        break

                if matched:
                    matched_genes.append(row)

            if matched_genes:
                filtered_df = pd.DataFrame(matched_genes)
                print(f"Found {len(filtered_df)} transcription factors after partial matching")
                return filtered_df

        filtered_df = gene_df[gene_df['name'].isin(tf_set)].copy()
        print(f"Found {len(filtered_df)} transcription factors from gene info")

        if len(filtered_df) < 10:
            print("Warning: Too few transcription factors after filtering, using all genes instead")
            return gene_df

        return filtered_df

    except Exception as e:
        print(f"Error filtering transcription factors: {e}")
        return gene_df


def extend_gene_regions(gene_df, upstream, downstream, include_gene_body):
    """Extend gene regions"""
    try:
        if include_gene_body:
            extended_df = pd.DataFrame()
            for _, row in gene_df.iterrows():
                new_row = row.copy()
                if row['strand'] == '+':
                    new_row['start'] = max(0, row['start'] - upstream)
                    new_row['end'] = row['end'] + downstream
                else:
                    new_row['start'] = max(0, row['start'] - downstream)
                    new_row['end'] = row['end'] + upstream

                extended_df = pd.concat([extended_df, pd.DataFrame([new_row])], ignore_index=True)

            print(f"Extended {len(extended_df)} gene regions (including gene body)")
            return extended_df
        else:
            extended_records = []
            for _, row in gene_df.iterrows():
                chrom = row['chrom']
                start = row['start']
                end = row['end']
                strand = row['strand']
                name = row['name']

                if strand == '+':
                    upstream_region = {
                        'chrom': chrom,
                        'start': max(0, start - upstream),
                        'end': start,
                        'name': f"{name}_upstream",
                        'score': 800,
                        'strand': strand
                    }
                    downstream_region = {
                        'chrom': chrom,
                        'start': end,
                        'end': end + downstream,
                        'name': f"{name}_downstream",
                        'score': 600,
                        'strand': strand
                    }
                    extended_records.extend([upstream_region, downstream_region])
                else:
                    upstream_region = {
                        'chrom': chrom,
                        'start': end,
                        'end': end + upstream,
                        'name': f"{name}_upstream",
                        'score': 800,
                        'strand': strand
                    }
                    downstream_region = {
                        'chrom': chrom,
                        'start': max(0, start - downstream),
                        'end': start,
                        'name': f"{name}_downstream",
                        'score': 600,
                        'strand': strand
                    }
                    extended_records.extend([upstream_region, downstream_region])

            extended_df = pd.DataFrame(extended_records)
            print(f"Extended {len(extended_df)} regions (upstream/downstream only, excluding gene body)")
            return extended_df

    except Exception as e:
        print(f"Error extending gene regions: {e}")
        return gene_df


def filter_peaks(peak_df, min_score):
    """Filter peaks by score"""
    if min_score <= 0:
        return peak_df

    try:
        peak_df['score'] = pd.to_numeric(peak_df['score'], errors='coerce')
        filtered_df = peak_df[peak_df['score'] >= min_score].copy()
        print(f"Retained {len(filtered_df)} peaks after score filtering")
        return filtered_df
    except Exception as e:
        print(f"Error filtering peaks: {e}")
        return peak_df


def intervals_overlap(a_start, a_end, b_start, b_end):
    """Check if two intervals overlap"""
    return a_start < b_end and b_start < a_end


def find_overlaps(peak_df, gene_df):
    """Find overlaps between peaks and gene regions"""
    try:
        print("Finding overlaps between peaks and gene regions...")

        chrom_to_genes = {}
        for _, gene in gene_df.iterrows():
            if gene['chrom'] not in chrom_to_genes:
                chrom_to_genes[gene['chrom']] = []
            chrom_to_genes[gene['chrom']].append(gene)

        peak_tf_mapping = {}
        overlap_count = 0
        tf_count = set()

        for _, peak in tqdm(peak_df.iterrows(), total=len(peak_df)):
            peak_chrom = peak['chrom']
            peak_start = peak['start']
            peak_end = peak['end']
            peak_id = f"{peak_chrom}-{peak_start}-{peak_end}"

            if peak_chrom not in chrom_to_genes:
                continue

            for gene in chrom_to_genes[peak_chrom]:
                if intervals_overlap(peak_start, peak_end, gene['start'], gene['end']):
                    tf_name = gene['name']

                    if '_upstream' in tf_name:
                        tf_name = tf_name.split('_upstream')[0]
                    elif '_downstream' in tf_name:
                        tf_name = tf_name.split('_downstream')[0]

                    if peak_id not in peak_tf_mapping:
                        peak_tf_mapping[peak_id] = set()
                    peak_tf_mapping[peak_id].add(tf_name)
                    tf_count.add(tf_name)
                    overlap_count += 1

        print(f"Found {overlap_count} peak-TF overlap relationships, involving {len(peak_tf_mapping)} peaks and {len(tf_count)} transcription factors")

        if len(peak_tf_mapping) == 0:
            print("Warning: No peak-TF overlap relationships found")
        else:
            print(f"Sample transcription factors found: {list(tf_count)[:10]}")

        return peak_tf_mapping

    except Exception as e:
        print(f"Error finding overlaps: {e}")
        return {}


def save_mapping_to_bed(peak_tf_mapping, output_file):
    """Save peak-TF mapping to BED file"""
    try:
        with open(output_file, 'w') as f:
            for peak_id, tf_set in peak_tf_mapping.items():
                if ':' in peak_id:
                    chrom, pos = peak_id.split(':')
                    start, end = pos.split('-')
                else:
                    parts = peak_id.split('-')
                    chrom = parts[0]
                    start = parts[1]
                    end = parts[2]

                tf_str = ','.join(sorted(tf_set))
                f.write(f"{chrom}\t{start}\t{end}\t{peak_id}\t{len(tf_set)}\t.\t{tf_str}\n")

        print(f"Peak-TF mapping saved as BED file: {output_file}")
        return True

    except Exception as e:
        print(f"Error saving BED file: {e}")
        return False


def save_mapping_to_csv(peak_tf_mapping, output_file):
    """Save peak-TF mapping to CSV file"""
    try:
        rows = []
        for peak_id, tf_set in peak_tf_mapping.items():
            if ':' in peak_id:
                chrom, pos = peak_id.split(':')
                start, end = pos.split('-')
            else:
                parts = peak_id.split('-')
                chrom = parts[0]
                start = parts[1]
                end = parts[2]

            for tf in tf_set:
                rows.append({
                    'peak_id': peak_id,
                    'chromosome': chrom,
                    'start': int(start),
                    'end': int(end),
                    'tf': tf
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)

        print(f"Peak-TF mapping saved as CSV file: {output_file}")
        return True

    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return False


def save_mapping_to_json(peak_tf_mapping, output_file):
    """Save peak-TF mapping to JSON file"""
    try:
        json_mapping = {peak_id: list(tf_set) for peak_id, tf_set in peak_tf_mapping.items()}

        with open(output_file, 'w') as f:
            json.dump(json_mapping, f, indent=2)

        print(f"Peak-TF mapping saved as JSON file: {output_file}")
        return True

    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False


def save_mapping(peak_tf_mapping, output_file, output_format):
    """Save mapping results in specified format"""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_format == 'bed':
        return save_mapping_to_bed(peak_tf_mapping, output_file)
    elif output_format == 'csv':
        return save_mapping_to_csv(peak_tf_mapping, output_file)
    elif output_format == 'json':
        return save_mapping_to_json(peak_tf_mapping, output_file)
    else:
        print(f"Unsupported output format: {output_format}")
        return False


def step3_create_mapping(args):
    """Step 3: Create peak-TF mapping relationships"""
    print("\n" + "=" * 60)
    print("Step 3: Create peak-TF mapping relationships")
    print("=" * 60)

    if not os.path.exists(args.peaks_input):
        print(f"Error: Peak file not found {args.peaks_input}")
        sys.exit(1)

    if not os.path.exists(args.gene_info_input):
        print(f"Error: Gene info file not found {args.gene_info_input}")
        sys.exit(1)

    # Load transcription factor list
    tf_set = None
    if args.tf_list and os.path.exists(args.tf_list):
        try:
            with open(args.tf_list, 'r') as f:
                tf_set = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(tf_set)} transcription factors from file")
        except Exception as e:
            print(f"Error loading transcription factor list: {e}")

    # Load peak file and gene info file
    print(f"Loading peak file: {args.peaks_input}")
    peak_df = load_bed_file(args.peaks_input)

    print(f"Loading gene info file: {args.gene_info_input}")
    gene_df = load_bed_file(args.gene_info_input)

    # Filter transcription factors
    if tf_set is not None:
        gene_df = filter_gene_info_for_tfs(gene_df, tf_set)

    # Extend gene regions
    print(f"Extending gene regions: upstream {args.upstream}bp, downstream {args.downstream}bp")
    gene_df = extend_gene_regions(
        gene_df,
        args.upstream,
        args.downstream,
        args.include_gene_body
    )

    # Filter peaks
    if args.min_score > 0:
        print(f"Filtering peaks by score, minimum score: {args.min_score}")
    peak_df = filter_peaks(peak_df, args.min_score)

    # Find overlaps
    peak_tf_mapping = find_overlaps(peak_df, gene_df)

    # Save results
    if peak_tf_mapping:
        success = save_mapping(peak_tf_mapping, args.mapping_output, args.output_format)
        if success:
            print(f"Step 3 completed: Mapped {len(peak_tf_mapping)} peaks to transcription factors")
            return True
        else:
            print("Failed to create peak-TF mapping file")
            sys.exit(1)
    else:
        print("Warning: No peak-TF overlap relationships found, no output file generated")
        sys.exit(1)


# =============================================================================
# Main Program
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Peak-TF Mapping Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
    # Run full pipeline
    python peak_tf_pipeline.py --mode all

    # Run step 1 only: Create gene info BED file
    python peak_tf_pipeline.py --mode create_gene_info

    # Run step 2 only: Extract ATAC peaks
    python peak_tf_pipeline.py --mode extract_peaks

    # Run step 3 only: Create peak-TF mapping
    python peak_tf_pipeline.py --mode create_mapping
        """
    )

    # Run mode
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'create_gene_info', 'extract_peaks', 'create_mapping'],
                        help='Run mode: all (full pipeline), create_gene_info (step 1), extract_peaks (step 2), create_mapping (step 3)')

    # -------------------------------------------------------------------------
    # Step 1 parameters: Create gene info BED file
    # -------------------------------------------------------------------------
    parser.add_argument('--gtf_input', type=str,
                        default='E:/code/mymodel2/data/Mouse_Brain/gencode.vM36.annotation.gff3',
                        help='Input GTF/GFF file path')
    parser.add_argument('--gene_info_output', type=str,
                        default='E:/code/mymodel2/data/Mouse_Brain/gene_info.bed',
                        help='Gene info BED file output path')
    parser.add_argument('--feature_type', type=str, default='gene',
                        help='Feature type to extract (e.g., gene, transcript)')
    parser.add_argument('--gene_id_attr', type=str, default='gene_id',
                        help='Gene ID attribute name')
    parser.add_argument('--gene_name_attr', type=str, default='gene_name',
                        help='Gene name attribute name')
    parser.add_argument('--promoter_upstream', type=int, default=2000,
                        help='Promoter region upstream distance (bp)')
    parser.add_argument('--promoter_downstream', type=int, default=500,
                        help='Promoter region downstream distance (bp)')
    parser.add_argument('--include_promoters', action='store_true',
                        help='Include promoter regions in output')

    # -------------------------------------------------------------------------
    # Step 2 parameters: Extract peak information from ATAC-seq data
    # -------------------------------------------------------------------------
    parser.add_argument('--atac_input', type=str,
                        default='E:/code/mymodel2/data/E18.5_mouse_brain/E18_adata_atac.h5ad',
                        help='ATAC-seq data file path (h5ad format)')
    parser.add_argument('--peaks_output', type=str,
                        default='E:/code/mymodel2/data/E18.5_mouse_brain/atac_peaks_me.bed',
                        help='Peak info BED file output path')
    parser.add_argument('--min_expression', type=float, default=0.0,
                        help='Minimum expression threshold, 0 for no filtering')
    parser.add_argument('--name_format', type=str, default='hyphen',
                        choices=['colon', 'underscore', 'hyphen'],
                        help='Peak name format: colon (chr1:1000-2000), underscore (chr1_1000_2000), or hyphen (chr1-1000-2000)')

    # -------------------------------------------------------------------------
    # Step 3 parameters: Create peak-TF mapping relationships
    # -------------------------------------------------------------------------
    parser.add_argument('--peaks_input', type=str,
                        default='E:/code/mymodel2/data/E18.5_mouse_brain/atac_peaks_me.bed',
                        help='ATAC-seq peak file (BED format)')
    parser.add_argument('--gene_info_input', type=str,
                        default='E:/code/mymodel2/data/Mouse_Brain/gene_info.bed',
                        help='Gene info file (BED format)')
    parser.add_argument('--tf_list', type=str,
                        default='E:/code/mymodel2/data/Mouse_Brain/trrust_rawdata.mouse.tsv',
                        help='Transcription factor list file')
    parser.add_argument('--mapping_output', type=str,
                        default='E:/code/mymodel2/data/E18.5_mouse_brain/peaks_to_tf_me.bed',
                        help='Peak-TF mapping output file path')
    parser.add_argument('--output_format', type=str, default='bed',
                        choices=['bed', 'csv', 'json'],
                        help='Output file format')
    parser.add_argument('--upstream', type=int, default=5000,
                        help='Upstream region length (bp)')
    parser.add_argument('--downstream', type=int, default=1000,
                        help='Downstream region length (bp)')
    parser.add_argument('--include_gene_body', action='store_true',
                        help='Include gene body region')
    parser.add_argument('--min_score', type=float, default=0,
                        help='Minimum peak score threshold')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    print("=" * 60)
    print("Peak-TF Mapping Analysis Pipeline")
    print("=" * 60)
    print(f"Run mode: {args.mode}")

    if args.mode == 'all':
        # Run full pipeline
        step1_create_gene_info(args)
        step2_extract_peaks(args)
        step3_create_mapping(args)
    elif args.mode == 'create_gene_info':
        step1_create_gene_info(args)
    elif args.mode == 'extract_peaks':
        step2_extract_peaks(args)
    elif args.mode == 'create_mapping':
        step3_create_mapping(args)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

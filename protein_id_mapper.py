import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re

class ProteinIDMapper:
    """
    蛋白质ID映射器，用于将ADT数据中的蛋白质名称映射到STRING数据库中的蛋白质ID
    """
    
    def __init__(self, manual_mapping=None, string_protein_info_file=None):
        """
        初始化蛋白质ID映射器
        
        参数:
            manual_mapping: 手动映射字典，格式为 {蛋白质名称: [基因ID, [蛋白质ID列表]]}
            string_protein_info_file: STRING数据库蛋白质信息文件路径，如9606.protein.info.v12.0.txt
        """
        # 手动映射现在为空，完全依赖STRING数据库的准确映射
        self.manual_mapping = manual_mapping or {}
        # self.manual_mapping = manual_mapping or {
        #     # 蛋白质名称: [基因ID, 蛋白质ID列表]
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
        #     # 补充剩余的蛋白质映射
        #     'CD45RO': ['ENSG00000081237', ['ENSP00000352288']],  # PTPRC变体
        #     'THY1': ['ENSG00000154096', ['ENSP00000367273']],    # CD90
        #     'FOXP3': ['ENSG00000049768', ['ENSP00000365380']],
        #     'ICOS': ['ENSG00000163600', ['ENSP00000370016']]
        # }
        
        
        # 从STRING数据库加载蛋白质信息
        self.string_protein_info = {}
        self.preferred_name_to_string_id = {}
        
        if string_protein_info_file and os.path.exists(string_protein_info_file):
            self.load_string_protein_info(string_protein_info_file)
        
        # 创建蛋白质名称到STRING ID的映射（现在主要依赖STRING数据库）
        self.protein_to_string_ids = {}
        for protein, mapping_data in self.manual_mapping.items():
            if isinstance(mapping_data, list) and len(mapping_data) >= 2:
                # 兼容原格式: [基因ID, [蛋白质ID列表]]
                gene_id, protein_ids = mapping_data[0], mapping_data[1]
                string_ids = protein_ids if isinstance(protein_ids, list) else [protein_ids]
                self.protein_to_string_ids[protein] = string_ids
        
        # 创建STRING ID到蛋白质名称的映射
        self.string_to_protein = {}
        for protein, string_ids in self.protein_to_string_ids.items():
            for string_id in string_ids:
                self.string_to_protein[string_id] = protein
    
    def load_string_protein_info(self, file_path):
        """
        从STRING数据库文件加载蛋白质信息
        只保留 string_protein_id 到 preferred_name 的映射
        
        标准格式: #string_protein_id	preferred_name	protein_size	annotation
        
        参数:
            file_path: STRING数据库蛋白质信息文件路径
        """
        print(f"从STRING数据库加载蛋白质信息: {file_path}")
        
        # 检测物种类型
        species_code = "unknown"
        if "9606" in file_path:
            species_code = "human"
        elif "10090" in file_path:
            species_code = "mouse"
        print(f"检测到物种: {species_code}")
        
        try:
            # STRING文件有固定的头部行格式: #string_protein_id	preferred_name	protein_size	annotation
            # 先读取头部行获取列名，然后读取数据
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()
                
            if header_line.startswith('#'):
                # 提取列名（去掉#号）
                column_names = header_line[1:].strip().split('\t')
                print(f"检测到头部行，列名: {column_names}")
                
                # 只读取前两列：string_protein_id 和 preferred_name
                df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None, 
                               names=column_names, usecols=[0, 1])
            else:
                # 没有头部行，使用默认列名，只读取前两列
                print("未检测到头部行，使用默认列名")
                df = pd.read_csv(file_path, sep='\t', header=None, 
                               names=['string_protein_id', 'preferred_name'], usecols=[0, 1])
            
            print(f"成功加载STRING蛋白质信息: {len(df)}行")
            print(f"实际列名: {list(df.columns)}")
            
        except Exception as e:
            print(f"读取文件失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 验证必需的列是否存在
        if 'string_protein_id' in df.columns and 'preferred_name' in df.columns:
            # 创建STRING ID到蛋白质信息的映射
            for _, row in df.iterrows():
                string_id = row['string_protein_id']
                preferred_name = row['preferred_name']
                
                # 标准化蛋白质ID格式（移除物种前缀如果存在）
                clean_string_id = string_id
                if '.' in string_id and (string_id.startswith('9606.') or string_id.startswith('10090.')):
                    clean_string_id = string_id.split('.', 1)[1]
                
                # 简化存储：只保存 preferred_name（使用清理后的ID作为主键）
                self.string_protein_info[clean_string_id] = preferred_name
                
                # 如果原始ID和清理后的ID不同，也存储原始ID的映射
                if string_id != clean_string_id:
                    self.string_protein_info[string_id] = preferred_name
                
                # 创建首选名称到STRING ID的映射（使用清理后的ID）
                if preferred_name not in self.preferred_name_to_string_id:
                    self.preferred_name_to_string_id[preferred_name] = []
                if clean_string_id not in self.preferred_name_to_string_id[preferred_name]:
                    self.preferred_name_to_string_id[preferred_name].append(clean_string_id)
            
            print(f"处理了{len(self.string_protein_info)}个蛋白质信息")
            print(f"创建了{len(self.preferred_name_to_string_id)}个首选名称到STRING ID的映射")
            
            # 打印一些示例
            sample_ids = list(self.string_protein_info.keys())[:5]
            print("\n示例蛋白质信息:")
            for string_id in sample_ids:
                preferred_name = self.string_protein_info[string_id]
                print(f"  {string_id}: {preferred_name}")
        else:
            print(f"错误: 文件格式不正确，列名: {list(df.columns)}")
    
    def extract_gene_symbols_from_annotation(self):
        """
        从首选名称构建基因符号映射（简化版本，不再使用注释）
        """
        print("从首选名称构建基因符号映射...")
        
        gene_symbol_to_string_ids = {}
        
        for string_id, preferred_name in self.string_protein_info.items():
            # 直接使用首选名称作为基因符号
            if preferred_name:
                if preferred_name not in gene_symbol_to_string_ids:
                    gene_symbol_to_string_ids[preferred_name] = []
                if string_id not in gene_symbol_to_string_ids[preferred_name]:
                    gene_symbol_to_string_ids[preferred_name].append(string_id)
        
        print(f"构建了{len(gene_symbol_to_string_ids)}个基因符号映射")
        
        # 更新映射
        self.gene_symbol_to_string_ids = gene_symbol_to_string_ids
        
        # 打印一些示例
        sample_symbols = list(gene_symbol_to_string_ids.keys())[:10]
        print("\n示例基因符号映射:")
        for symbol in sample_symbols:
            string_ids = gene_symbol_to_string_ids[symbol][:3]  # 最多显示3个
            print(f"  {symbol}: {string_ids}")
        
        return gene_symbol_to_string_ids
    
    def map_protein_to_string_ids(self, protein_name):
        """
        将蛋白质名称映射到STRING ID
        
        参数:
            protein_name: 蛋白质名称
            
        返回:
            STRING ID列表，如果未找到则返回空列表
        """
        # 1. 优先检查STRING数据库中的首选名称映射（最准确）
        if protein_name in self.preferred_name_to_string_id:
            return self.preferred_name_to_string_id[protein_name]
        
        # 2. 检查基因符号映射
        if hasattr(self, 'gene_symbol_to_string_ids') and protein_name in self.gene_symbol_to_string_ids:
            return self.gene_symbol_to_string_ids[protein_name]
        
        # 3. 备选：检查手动映射（可能过时）
        if protein_name in self.protein_to_string_ids:
            return self.protein_to_string_ids[protein_name]
        
        # 4. 模糊匹配 - 检查首选名称是否包含蛋白质名称
        matches = []
        for name, string_ids in self.preferred_name_to_string_id.items():
            if protein_name.lower() in name.lower() or name.lower() in protein_name.lower():
                matches.extend(string_ids)
        
        if matches:
            return matches
        
        return []
    
    def map_string_id_to_protein(self, string_id):
        """
        将STRING ID映射到蛋白质名称
        
        参数:
            string_id: STRING ID
            
        返回:
            蛋白质名称，如果未找到则返回None
        """
        # 1. 检查手动映射
        if string_id in self.string_to_protein:
            return self.string_to_protein[string_id]
        
        # 2. 检查STRING蛋白质信息（现在直接存储preferred_name）
        if string_id in self.string_protein_info:
            return self.string_protein_info[string_id]
        
        return None
    
    def find_protein_in_ppi_network(self, protein_name, ppi_network):
        """
        在PPI网络中查找蛋白质
        
        参数:
            protein_name: 蛋白质名称
            ppi_network: PPI网络，格式为 {string_id: {string_id: score}}
            
        返回:
            (是否找到, 对应的STRING ID列表)
        """
        # 1. 使用映射方法查找STRING ID
        string_ids = self.map_protein_to_string_ids(protein_name)
        if string_ids:
            # 检查这些ID是否在PPI网络中
            found_ids = [sid for sid in string_ids if sid in ppi_network]
            if found_ids:
                return True, found_ids
        
        # 2. 尝试直接匹配
        if protein_name in ppi_network:
            return True, [protein_name]
        
        # 3. 模糊匹配 - 检查STRING ID是否包含蛋白质名称
        for string_id in ppi_network.keys():
            # 去掉物种前缀
            if '.' in string_id:
                base_id = string_id.split('.')[1]
            else:
                base_id = string_id
                
            # 检查是否包含蛋白质名称
            if protein_name.lower() in base_id.lower():
                return True, [string_id]
        
        # 4. 更宽松的模糊匹配 - 使用前缀
        if len(protein_name) > 3:
            prefix = protein_name[:3].lower()
            for string_id in ppi_network.keys():
                # 去掉物种前缀
                if '.' in string_id:
                    base_id = string_id.split('.')[1]
                else:
                    base_id = string_id
                    
                # 检查是否包含蛋白质名称前缀
                if prefix in base_id.lower():
                    return True, [string_id]
        
        return False, []
    
    def create_mapping_for_adt_data(self, adt_proteins, ppi_network=None, gene_to_ensembl=None):
        """
        为ADT数据中的蛋白质创建映射
        
        参数:
            adt_proteins: ADT数据中的蛋白质名称列表
            ppi_network: 可选，PPI网络，格式为 {string_id: {string_id: score}}
            gene_to_ensembl: 可选，基因符号到Ensembl ID的映射
            
        返回:
            映射字典，格式为 {蛋白质名称: {'source': 来源, 'gene_id': 基因ID, 'string_ids': STRING ID列表}}
        """
        # 如果有STRING蛋白质信息但没有提取基因符号，先提取
        if self.string_protein_info and not hasattr(self, 'gene_symbol_to_string_ids'):
            self.extract_gene_symbols_from_annotation()
        
        mapping = {}
        
        for protein in adt_proteins:
            # 1. 使用手动映射
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
            
            # 2. 使用STRING数据库映射
            string_ids = self.map_protein_to_string_ids(protein)
            if string_ids:
                # 尝试找到基因ID
                gene_id = None
                if gene_to_ensembl and protein in gene_to_ensembl:
                    gene_id = gene_to_ensembl[protein]
                
                mapping[protein] = {
                    'source': 'string_database',
                    'gene_id': gene_id,
                    'string_ids': string_ids
                }
                continue
            
            # 3. 使用PPI网络查找
            if ppi_network:
                found, string_ids = self.find_protein_in_ppi_network(protein, ppi_network)
                if found:
                    mapping[protein] = {
                        'source': 'ppi_network',
                        'gene_id': None,  # PPI网络中通常没有基因ID
                        'string_ids': string_ids
                    }
                    continue
            
            # 4. 使用基因符号到Ensembl ID的映射
            if gene_to_ensembl and protein in gene_to_ensembl:
                gene_id = gene_to_ensembl[protein]
                # 从基因ID猜测蛋白质ID
                protein_id = gene_id.replace('ENSG', 'ENSP')
                string_id = protein_id
                mapping[protein] = {
                    'source': 'gene_mapping',
                    'gene_id': gene_id,
                    'string_ids': [string_id]
                }
                continue
            
            # 5. 未找到映射
            mapping[protein] = {
                'source': 'not_found',
                'gene_id': None,
                'string_ids': []
            }
        
        return mapping
    
    def print_mapping_stats(self, mapping):
        """
        打印映射统计信息
        
        参数:
            mapping: 映射字典
        """
        sources = {}
        for protein, info in mapping.items():
            source = info['source']
            if source not in sources:
                sources[source] = []
            sources[source].append(protein)
        
        print("\n映射统计:")
        for source, proteins in sources.items():
            print(f"{source}: {len(proteins)} 个蛋白质")
            if len(proteins) > 0:
                print(f"  示例: {proteins[:min(3, len(proteins))]}")
        
        # 计算总的映射率
        total = len(mapping)
        mapped = total - len(sources.get('not_found', []))
        print(f"\n总映射率: {mapped}/{total} ({mapped/total:.2%})")

def main():
    """主函数，用于测试蛋白质ID映射器"""
    # 设置STRING数据库文件路径
    string_protein_info_file = 'data/Data_SpatialGlue/Dataset1_Mouse_Spleen1/10090.protein.info.v12.0.txt'
    
    # 创建映射器
    mapper = ProteinIDMapper(string_protein_info_file=string_protein_info_file)
    
    # 测试ADT蛋白质列表
    adt_proteins = [
        'CD163', 'CR2', 'PCNA', 'VIM', 'KRT5', 'CD68', 'CEACAM8', 'PTPRC', 'HLA-DRA', 'PAX5',
        'SDC1', 'CD8A', 'BCL2', 'CD19', 'PDCD1', 'ACTA2', 'FCGR3A', 'ITGAX', 'CXCR5', 'EPCAM',
        'MS4A1', 'CD3E', 'CD14', 'CD40', 'PECAM1', 'CD4', 'ITGAM', 'CD27', 'CCR7', 'CD274'
    ]
    
    # 创建映射
    mapping = mapper.create_mapping_for_adt_data(adt_proteins)
    
    # 打印映射统计
    mapper.print_mapping_stats(mapping)
    
    # 打印详细映射
    print("\n详细映射:")
    for protein, info in mapping.items():
        if info['source'] != 'not_found':
            print(f"{protein}: {info['string_ids']}")

if __name__ == "__main__":
    main() 
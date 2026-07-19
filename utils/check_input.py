# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""
from Bio import SeqIO

def check_input(file_list):
    checked_list = []
    for seq_file in file_list:
        if not seq_file.endswith('.fasta'):
            print(f'{seq_file} file_name error')
            continue
        
        count_seq = 0
        have_error = False
        chains_list = []
        for record in SeqIO.parse(seq_file, 'fasta'):
            try:
                chain_seq = str(record.seq).strip()
                header = str(record.name).strip()
                chain_id, mol_type = header.split('|')
                chains_list.append(chain_id)
            except:
                print(f'{seq_file} parse error')
                have_error = True
                continue
            
            if mol_type not in ['dna', 'rna']:
                print(f'{seq_file} mol_type error')
                have_error = True
            
            if not set(chain_seq).issubset(set('AUCGTX')):
                print(f'{seq_file} seq error')
                have_error = True           
            
            count_seq += 1
            
        if have_error:
            continue
        
        if len(chains_list)>len(set(chains_list)):
            print(f'{seq_file} chain_id error')
            continue     
            
        if count_seq==1:
            checked_list.append(seq_file)
        else:
            print(f'{seq_file} count_seq error')
            continue  
    
    return checked_list
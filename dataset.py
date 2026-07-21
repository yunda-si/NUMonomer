# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

from torch.utils.data import Dataset
import torch
from pathlib import Path
import np.residue_constants as resc
import numpy as np
from Bio import SeqIO


class MonomerDataset(Dataset):

    def __init__(self,
                 targets_list=None,
                 num_structure_recycle=8,
                 clamp_plddt=768,
                 mask_seq=False,
                 atom_type_num=28,
                 ):

        self.atom_type_num = atom_type_num
        self.targets_list = targets_list
        self.num_structure_recycle = num_structure_recycle
        self.clamp_plddt = clamp_plddt
        self.mask_seq = mask_seq
        self.c3_idx = resc.atom_types.index("C3'")
        
    def __getitem__(self, idx):

        seq_file = self.targets_list[idx]
        target_name = Path(seq_file).stem

        monomer_feature = {'target_name':target_name,
                           'sel_idx': None,
                           'mol_type':None,
                           'atom_coords':None,
                           'label_seq':None,
                           'chain_list':None,
                           'num_structure_recycle':self.num_structure_recycle,
                           'clamp_plddt':self.clamp_plddt
                           }
        
        # seq
        record = [i for i in SeqIO.parse(seq_file, 'fasta')][0]
        chain_seq = str(record.seq).strip()
        header = str(record.name)
        chain_id, mol_type = header.strip().split('|')

        seq_np = [resc.restype1_order[i] for i in chain_seq]
        seq_length = len(chain_seq)
        sel_idx = torch.arange(seq_length)

        monomer_feature['sel_idx'] = sel_idx
        monomer_feature['mol_type'] = mol_type
        monomer_feature['label_seq'] = torch.from_numpy(np.array(seq_np)).long().unsqueeze(0)
        monomer_feature['chain_list'] = chain_id
        
        # pdb
        peptide_coords = [self.buildpeptide(seq_length)]
        peptide_coords = torch.stack(peptide_coords)
        peptide_coords = peptide_coords / 10
        monomer_feature['na_coords'] = peptide_coords.permute(0, 2, 1, 3)

        return monomer_feature

    def buildpeptide(self, seq_length):

        pep_coords = torch.rand(seq_length, self.atom_type_num, 3).float()
        pep_coords -= torch.mean(pep_coords[:,:3,:], dim=(0,1))

        return pep_coords

    def __len__(self):
        return len(self.targets_list)

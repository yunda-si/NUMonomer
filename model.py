# -*- coding: utf-8 -*-
"""
@author: yunda_si
"""

import torch.nn as nn
import torch
from modules import InputEmbedder, ConfidenceHead
from structure_module import StructureModule
from utils.seq_utils import seq2atom, localrigids
from utils.rigid_utils import rot_to_quat

def log_memory():
    print("   torch.cuda.memory_allocated: %8.4f GB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("    torch.cuda.memory_reserved: %8.4f GB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %8.4f GB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    
    
class NUMonomer(nn.Module):

    def __init__(self,
                 blocks_structure_decoder,
                 blocks_confidence,
                 atom_channel,
                 atom_nhead,
                 pair_channel,
                 pair_nhead,
                 tm_bin=64,
                 num_atom=28,
                 plddt_bin=50,
                 dropout_p=0.0,
                 dropout_p2d=0.0,
                 split_seq=None,
                 split_atom=None,
                 ):

        super(NUMonomer, self).__init__()

        self.dropout_p = dropout_p
        self.num_atom = num_atom
        self.pair_channel = pair_channel
        self.split_seq = split_seq

        self.embed_struc = nn.Embedding(num_atom + 1, atom_channel)

        self.input_embedder = InputEmbedder(pair_channel=pair_channel,
                                            nums_aa=6,
                                            atom_channel=atom_channel)

        self.trans_seq = nn.Sequential(nn.RMSNorm(atom_channel)) 

        self.structure_block = StructureModule(atom_channel=atom_channel,
                                               pair_channel=pair_channel,
                                               atom_nhead=atom_nhead,
                                               block_num=blocks_structure_decoder,
                                               dropout_p=dropout_p,
                                               num_atom=num_atom,
                                               split_atom=split_atom,
                                               split_seq=split_seq)

        self.confidence_block = ConfidenceHead(nblock=blocks_confidence,
                                               atom_channel=atom_channel,
                                               atom_nhead=atom_nhead,
                                               pair_channel=pair_channel,
                                               pair_nhead=pair_nhead,
                                               no_bins_lddt=plddt_bin,
                                               nums_aa=6,
                                               dropout_p=dropout_p,
                                               dropout_p2d=dropout_p2d,
                                               split_seq=split_seq,
                                               )

    def forward(self, monomer, last_only):

        pred_coords = []
        pred_plddts = []
        if last_only:
            plddt_iter = [monomer['num_structure_recycle'] - 1]
        else:
            plddt_iter = [i for i in range(monomer['num_structure_recycle'])]
        
        pair_idx, emb_params, seq_init = self.input_embedder(monomer)

        coords = monomer['na_coords']
        basis,_ = localrigids(torch.transpose(coords, 1, 2))
        quat = rot_to_quat(basis)

        struc_emb = seq2atom(monomer['label_seq'], monomer['mol_type'])
        struc_emb = self.embed_struc(struc_emb) + seq_init
        

        for i_struc_cycle in range(monomer['num_structure_recycle']):
            if i_struc_cycle > 0:
                struc_emb = self.trans_seq(struc_emb)

            coords, struc_emb, quat = self.structure_block(struc_emb, pair_idx, emb_params, coords, quat)
            
            if i_struc_cycle in plddt_iter:
                pred_coords.append(coords * 10)
                plddt = self.confidence_block(monomer['label_seq'][0:1,:monomer['clamp_plddt']], 
                                              coords[0:1,:,:monomer['clamp_plddt']]*10, 
                                              pair_idx[:monomer['clamp_plddt'],:monomer['clamp_plddt']],
                                              emb_params)
                pred_plddts.append(plddt)

        pred_coords = torch.permute(torch.stack(pred_coords), [0, 1, 3, 2, 4])
        pred_plddts = torch.stack(pred_plddts)

        log_memory()
        
        return {
                'pred_coords': pred_coords,
                'pred_plddts': pred_plddts,
                }
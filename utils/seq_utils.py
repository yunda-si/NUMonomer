# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

import torch
from np import residue_constants as resc

def localrigids(coords37,
                n_idx=resc.atom_order["P"],
                ca_idx=resc.atom_order["C3'"],
                c_idx=resc.atom_order["C1'"],
                eps=1e-8,
                ):

    n_coords = coords37[..., n_idx, :]
    ca_coords = coords37[..., ca_idx, :]
    c_coords = coords37[..., c_idx, :]

    v1 = ca_coords - n_coords
    v2 = c_coords - ca_coords
    e1 = v1 / (eps + torch.linalg.norm(v1, dim=-1, keepdim=True))
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = u2 / (eps + torch.linalg.norm(u2, dim=-1, keepdim=True))
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.concat((e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)), dim=-1)
    T = ca_coords

    return R, T

def make_atom28_masks(na_aatype, na_type):
    residx_atom28_mask = resc.restype_atom28_mask[[resc.restype3_order[resc.restype_1to3[na_type][i]] if i in resc.restype_1to3[na_type].keys() else -1 for i in na_aatype]]
    residx_atom28_idx = []
    for row in residx_atom28_mask:
        sel_idx = [idx for idx, data in enumerate(row) if data > 0.4]
        if len(sel_idx) > 3:
            residx_atom28_idx.append(sel_idx)
        else:
            residx_atom28_idx.append([i for i in range(resc.atom_type_num)])

    return residx_atom28_mask, residx_atom28_idx

res_atom_dict = {}
for res in resc.restypes3:
    temp = torch.zeros(resc.atom_type_num) # 0 means None
    for atom in resc.residue_atoms[res]:
        temp[resc.atom_order[atom]] = resc.atom_order[atom]+1
    res_atom_dict[res] = temp.long()


def seq2atom(num_seqs, seq_type='dna'):

    atom_mat = torch.zeros(num_seqs.shape[0], resc.atom_type_num, num_seqs.shape[1], device=num_seqs.device)
    if type(num_seqs) == torch.Tensor:
        num_seqs = num_seqs.detach().cpu().numpy()

    for idx_seq in range(num_seqs.shape[0]):
        for idx_res in range(num_seqs.shape[1]):
            res = int(num_seqs[idx_seq, idx_res])
            if res in resc.restypes1:
                atom_mat[idx_seq, :, idx_res] = res_atom_dict[resc.restype_1to3[seq_type][resc.rev_restype1_order[res]]]

    return atom_mat.long()


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100
    
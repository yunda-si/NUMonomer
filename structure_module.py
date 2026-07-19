#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yunda_si
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from utils.rigid_utils import quat_to_rot, quat_multiply_by_vec
from config import get_cfg
cfg = get_cfg()


def ipa(q, k, scale_q, qp, vp, kp, coords, basis, pair_bias, head_weights):
    q_attn_weights = torch.einsum('baihc,bajhc -> bhaij', q * scale_q, k)

    qp_attn_weights = torch.einsum('bhlipc ->bhli', qp ** 2).unsqueeze(-2) + torch.einsum('bhlipc ->bhli',
                                                                                          kp ** 2).unsqueeze(
        -1) - 2 * torch.einsum('bhlipc, bhljpc ->bhlji', qp, kp)

    attn = (pair_bias + q_attn_weights + qp_attn_weights*head_weights) * (3 ** -0.5)

    o3 = torch.einsum('bhaij, bhajpc -> bhaipc', attn.softmax(-1), vp) - coords.unsqueeze(1).unsqueeze(-2)
    o3 = torch.einsum('bhalpi, blij -> balhpj', o3, basis)

    return o3

def ipaa(q, k, scale_q, qp, vp, kp, coords, basis, head_weights):
    q_attn_weights = torch.einsum('bilhc,bjlhc -> bhlij', q * scale_q, k)

    qp_attn_weights = torch.einsum('bhlipc ->bhli', qp ** 2).unsqueeze(-2) + torch.einsum('bhlipc ->bhli',
                                                                                          kp ** 2).unsqueeze(
        -1) - 2 * torch.einsum('bhlipc, bhljpc ->bhlji', qp, kp)

    attn = (q_attn_weights + qp_attn_weights * head_weights) * (2 ** -0.5)

    o3 = torch.einsum('bhlij, bhljpc -> bhlipc', attn.softmax(-1), vp) - coords.unsqueeze(1).unsqueeze(-2)
    o3 = torch.einsum('bhlapi, blij -> balhpj', o3, basis).flatten(-3, -1)

    return o3


comp_ipa = torch.compile(ipa, fullgraph=False, dynamic=True, disable=cfg.inference.disable_compile)
comp_ipaa = torch.compile(ipaa, fullgraph=False, dynamic=True, disable=cfg.inference.disable_compile)
comp_einsum = torch.compile(torch.einsum, fullgraph=False, dynamic=True, disable=cfg.inference.disable_compile)
comp_silu = torch.compile(nn.SiLU(), fullgraph=False, dynamic=True, disable=cfg.inference.disable_compile)


class StructureModuleTransition(nn.Module):

    def __init__(
                self,
                in_channels,
                dropout_p=0.1,
                scalen=4,
                ):
        super().__init__()

        hidden_channels = scalen*in_channels

        self.norm = nn.RMSNorm(in_channels)
        self.left = nn.Linear(in_channels, hidden_channels, bias=False)
        self.right = nn.Linear(in_channels, hidden_channels, bias=False)
        self.linear_ff = nn.Linear(hidden_channels, in_channels, bias=False)

        self.dropout_module = nn.Dropout(dropout_p)
        self.act = comp_silu

    def forward(self, x):

        x = self.norm(x)

        x = self.linear_ff(self.act(self.left(x)) * self.right(x))

        return self.dropout_module(x)


class IPA(nn.Module):

    def __init__(self,
                 pair_channel=128,
                 atom_channel=128,
                 num_atom=27,
                 atom_head=8,
                 points=8,
                 eps=1e-6,
                 dropout_p=0.0,
                 split_seq=None,
                 ):

        super(IPA, self).__init__()

        self.nhead = atom_head
        self.head_dim = atom_channel // atom_head
        self.scale_q = self.head_dim ** (-0.5)
        self.points = points
        self.scale_qp = (points * 9.0 / 2) ** (-0.5)
        self.eps = eps
        self.num_atom = num_atom
        self.split_seq = split_seq

        self.ln_seq_in = nn.RMSNorm(atom_channel)

        self.qk_seq = nn.Linear(atom_channel, atom_channel * 2, bias=False)
        self.qkvp_seq = nn.Linear(atom_channel, atom_head * points * 3 * 3, bias=False)

        self.norm_pair = nn.RMSNorm(pair_channel)
        self.trans_pair = nn.Parameter((torch.rand(1, pair_channel, atom_head) - 0.5) * 2 / (pair_channel ** 0.5))
        self.head_weights = nn.Parameter(torch.zeros(self.nhead, 1, 1, 1))

        self.linear_out = nn.Linear(atom_head * (points * 3), atom_channel)

        self.softplus = nn.Softplus()
        self.dropout_attn = nn.Dropout(dropout_p)
        self.dropout_module = nn.Dropout(dropout_p)


    def forward(self, seq, idx_mat, pair_params, coords, basis):

        seq = self.ln_seq_in(seq)

        batch, num_atom, num_residue, _ = seq.shape

        q = self.qk_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim*2)
        q, k = torch.split(q, self.head_dim, -1)

        qp = self.qkvp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points*3, 3)
        qp = comp_einsum('balhpi, blji -> bhalpj', qp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        qp, kp, vp = torch.split(qp, self.points, -2)

        head_weights = self.softplus(self.head_weights) * (-0.5*self.scale_qp)
        pair_params = comp_einsum('bc, ach -> bha', self.norm_pair(pair_params), self.trans_pair).flatten(-2, -1)
            
        if self.split_seq != None:
            o3 = []
            for res_idx in torch.split(torch.arange(seq.shape[2]), self.split_seq):
                pair_bias = torch.embedding(pair_params, idx_mat[res_idx,:].int()).unsqueeze(0).unflatten(-1, (self.nhead, 1)).permute([0,3,4,1,2])
                
                temp = comp_ipa(q[:,:,res_idx], 
                                k, 
                                self.scale_q, 
                                qp, 
                                vp,
                                kp[:,:,:,res_idx],
                                coords[:,:,res_idx], 
                                basis[:,res_idx], 
                                pair_bias, 
                                head_weights)
                o3.append(temp)
            o3 = torch.concat(o3, dim=2).flatten(-3, -1)  
                
                
        else:
            pair_bias = torch.embedding(pair_params, idx_mat.int()).unsqueeze(0).unflatten(-1, (self.nhead, 1)).permute([0,3,4,1,2])
            o3 = comp_ipa(q, k, self.scale_q, qp, vp, kp, coords, basis, pair_bias, head_weights).flatten(-3, -1)  

        return self.dropout_module(self.linear_out(o3))


class IPAATOM(nn.Module):

    def __init__(self,
                 atom_channel=128,
                 pair_channel=128,
                 num_atom=26,
                 atom_head=8,
                 points=8,
                 eps=1e-6,
                 dropout_p=0.0,
                 ):
        super(IPAATOM, self).__init__()

        self.nhead = atom_head
        self.head_dim = atom_channel // atom_head
        self.scale_q = self.head_dim ** (-0.5)
        self.points = points
        self.scale_qp = (points * 9.0 / 2) ** (-0.5)
        self.eps = eps

        self.ln_seq_in = nn.RMSNorm(atom_channel)

        self.qk_seq = nn.Linear(atom_channel, atom_channel*2, bias=False)

        self.qkvp_seq = nn.Linear(atom_channel, atom_head * points * 3 * 3, bias=False)

        self.head_weights = nn.Parameter(torch.zeros(self.nhead, 1, 1, 1))

        self.linear_out = nn.Linear(atom_head * (points * 3), atom_channel)

        self.softplus = nn.Softplus()
        self.dropout_attn = nn.Dropout(dropout_p)
        self.dropout_module = nn.Dropout(dropout_p)


    def forward(self, seq, coords, basis):

        coords = torch.permute(coords, [0, 2, 1, 3]) #blac

        seq = self.ln_seq_in(seq)
        batch, num_atom, num_residue, _ = seq.shape

        q = self.qk_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim*2)
        q, k = torch.split(q, self.head_dim, -1)

        qp = self.qkvp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points*3, 3)
        qp = comp_einsum('balhpi, blji -> bhlapj', qp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        qp, kp, vp = torch.split(qp, self.points,-2)

        head_weights = self.softplus(self.head_weights)* (-0.5 * self.scale_qp)

        o3 = comp_ipaa(q, k, self.scale_q, qp, vp, kp, coords, basis, head_weights)

        return self.dropout_module(self.linear_out(o3))


class StructureBlock(nn.Module):
    def __init__(self,
                 atom_channel,
                 pair_channel,
                 atom_nhead,
                 num_atom=28,
                 dropout_p=0.0,
                 points=8,
                 eps=1e-7,
                 split_seq=None,
                 split_atom=None,
                 ):

        super(StructureBlock, self).__init__()

        self.num_atom = num_atom
        self.split_seq = split_seq
        self.split_atom = split_atom

        self.ipa = IPA(pair_channel=pair_channel,
                       atom_channel=atom_channel,
                       num_atom=num_atom,
                       atom_head=atom_nhead,
                       points=points,
                       eps=eps,
                       dropout_p=dropout_p,
                       split_seq=split_seq,
                       )

        self.ipa_atom = IPAATOM(atom_channel=atom_channel,
                                pair_channel=pair_channel,
                                num_atom=num_atom,
                                atom_head=atom_nhead,
                                points=points,
                                eps=eps,
                                dropout_p=dropout_p
                                )

        self.transition = StructureModuleTransition(in_channels=atom_channel,
                                                    dropout_p=dropout_p)

    def forward(self, struc_emb, pair, emb_params, coords, basis):

        if self.split_atom != None:
            for atom_idx in torch.split(torch.arange(self.num_atom), self.split_atom):
                struc_emb[:, atom_idx] += self.ipa(struc_emb[:, atom_idx], pair, emb_params, coords[:, atom_idx], basis)
        else:
            struc_emb += self.ipa(struc_emb, pair, emb_params, coords, basis)

        if self.split_seq != None:
            for res_idx in torch.split(torch.arange(struc_emb.shape[2]), self.split_seq):
                struc_emb[:, :, res_idx] += self.ipa_atom(struc_emb[:, :, res_idx], coords[:, :, res_idx], basis[:, res_idx])
        else:
          struc_emb += self.ipa_atom(struc_emb, coords, basis)


        if self.split_seq != None:
            for res_idx in torch.split(torch.arange(struc_emb.shape[2]), self.split_seq):
                struc_emb[:, :, res_idx] += self.transition(struc_emb[:, :, res_idx])
        else:
            struc_emb += self.transition(struc_emb)

        return struc_emb


class StructureModule(nn.Module):

    def __init__(self,
                 atom_channel,
                 pair_channel,
                 atom_nhead,
                 num_atom=28,
                 block_num=8,
                 using_flash=True,
                 dropout_p=0.0,
                 split_seq=None,
                 split_atom=None,
                 ):

        super(StructureModule, self).__init__()

        self.dropout_p = dropout_p
        self.using_flash = using_flash
        self.split_atom = split_atom
        self.split_seq = split_seq

        self.structure_block = self._make_atom_encoder(block_num, atom_channel, pair_channel, atom_nhead, num_atom)
        self.ln_seq_out = nn.RMSNorm(atom_channel)

        self.coords_out = nn.Sequential(nn.Linear(atom_channel, atom_channel, bias=True),
                                        nn.RMSNorm(atom_channel),
                                        torch.compile(nn.GELU(), disable=cfg.inference.disable_compile))

        self.weights_coords = nn.Parameter((torch.rand(num_atom, atom_channel, 3) - 0.5) * 2 / (atom_channel ** 0.5))

        self.basis_out = nn.Sequential(nn.Linear(atom_channel, atom_channel, bias=True),
                                       nn.RMSNorm(atom_channel),
                                       torch.compile(nn.GELU(), disable=cfg.inference.disable_compile),
                                       nn.Linear(atom_channel, 3, bias=False),
                                       )

    def _make_atom_encoder(self, block_num, atom_channel, pair_channel, atom_nhead, num_atom):

        layers = []
        for index in range(block_num):
            layer = StructureBlock(atom_channel=atom_channel,
                                   pair_channel=pair_channel,
                                   atom_nhead=atom_nhead,
                                   num_atom=num_atom,
                                   dropout_p=self.dropout_p,
                                   split_atom=self.split_atom,
                                   split_seq=self.split_seq)

            layers.append(('struc_block' + str(index), layer))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, struc_emb, pair, emb_params, ori_coords, quat, c3p_idx=5):

        basis = quat_to_rot(quat)

        for idx_layer, layer in enumerate(self.structure_block):
            struc_emb = layer(struc_emb, pair, emb_params, ori_coords, basis)

        seq = self.ln_seq_out(struc_emb)
        coords = ori_coords + comp_einsum('bali, aij, blkj -> balk', self.coords_out(seq), self.weights_coords, basis)

        quat_update = self.basis_out(seq[:, c3p_idx, :, :])
        new_quat = quat + quat_multiply_by_vec(quat, quat_update)
        new_quat = new_quat / torch.linalg.norm(new_quat, dim=-1, keepdim=True)

        return coords, struc_emb, new_quat



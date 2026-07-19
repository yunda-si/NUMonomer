# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from config import get_cfg
cfg = get_cfg()


comp_einsum = torch.compile(torch.einsum, disable=cfg.inference.disable_compile)
comp_sigmoid = torch.compile(torch.sigmoid, disable=cfg.inference.disable_compile)
comp_silu = torch.compile(nn.SiLU(), disable=cfg.inference.disable_compile)


if cfg.inference.use_dsattn:
    from flash_attn import flash_attn_func
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention as DSAttn

class Attn(nn.Module):
    def __init__(self,
                 in_channels,
                 nhead,
                 dropout_p=0.1,
                 is_causal=False,
                 tied_attn=False
                 ):

        super(Attn, self).__init__()


        self.in_channels = in_channels
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.head_dim = self.in_channels // self.nhead
        self.scaling = self.head_dim ** -0.5
        self.is_causal = is_causal
        self.tied_attn = tied_attn
        self.attn_shape = 'bhij'

        self.norm = nn.RMSNorm(in_channels)
        self.linear_q = nn.Linear(in_channels, in_channels, bias=False)
        self.linear_k = nn.Linear(in_channels, in_channels, bias=False)
        self.linear_v = nn.Linear(in_channels, in_channels, bias=False)
        self.linear_ff = nn.Linear(in_channels, in_channels, bias=True)
        self.gate = nn.Linear(in_channels, in_channels, bias=True)

        self.dropout_module = nn.Dropout(dropout_p)
        self.dropout_attn = nn.Dropout(dropout_p)
        self.sigmoid = comp_sigmoid


    def cal_attn(self, q, k, v, scale, attn_bias):

        q *= scale
        attn_weights = comp_einsum(f'bhic,bhjc -> {self.attn_shape}', q, k)
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

        attn_prob = attn_weights.softmax(-1)
        attn_prob = self.dropout_attn(attn_prob)

        output = torch.einsum(f'{self.attn_shape}, bhjc -> bhic', attn_prob, v)

        return output

    def forward(self, x, attn_bias=None):

        batch, num_row, num_column, _ = x.shape
        comb_batch = batch*num_row

        x = torch.flatten(x,0,1)
        if attn_bias is not None:
            attn_bias = torch.flatten(attn_bias,0,1)
   
        x = self.norm(x)

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if attn_bias is not None:
            q = q * (2 ** -0.5)
            attn_bias = attn_bias * (2 ** -0.5)

        q = q.view(comb_batch, num_column, self.nhead, self.head_dim)
        k = k.view(comb_batch, num_column, self.nhead, self.head_dim)
        v = v.view(comb_batch, num_column, self.nhead, self.head_dim)

        if cfg.inference.use_dsattn and num_row>24 and num_column>24:
            if attn_bias is not None:
                output = DSAttn(q.unflatten(0,(batch,num_row)), k.unflatten(0,(batch,num_row)), v.unflatten(0,(batch,num_row)), [None, attn_bias.unflatten(0,(batch,1))])
                output = output.flatten(0,1)
            else:
                output = flash_attn_func(q, k, v,
                                         softmax_scale=self.scaling,
                                         dropout_p=0.0,
                                         window_size=self.window_size,
                                         causal=self.is_causal)
        else:
            q = q.transpose(1, 2).unflatten(0,(batch,num_row))
            k = k.transpose(1, 2).unflatten(0,(batch,num_row))
            v = v.transpose(1, 2).unflatten(0,(batch,num_row))
            output = F.scaled_dot_product_attention(q, k, v,
                                                    scale=self.scaling,
                                                    dropout_p=0.0,
                                                    attn_mask=attn_bias.unflatten(0,(batch,1)) if attn_bias is not None else attn_bias,
                                                    is_causal=self.is_causal)
            output = output.flatten(0,1).transpose(1, 2)
    
        g = self.sigmoid(self.gate(x))
        g = g.view(comb_batch, num_column, self.nhead, self.head_dim)
        
        output = (g * output).view(comb_batch, num_column, self.in_channels)

        output = self.dropout_module(self.linear_ff(output))
        output = torch.unflatten(output,0,(batch, num_row))

        return output


class FeedForwardNetwork(nn.Module):

    def __init__(
                self,
                in_channels,
                dropout_p=0.1,
                scalen=4,
                ):
        super(FeedForwardNetwork, self).__init__()

        hidden_channels = scalen*in_channels

        self.norm = nn.RMSNorm(in_channels)
        self.left = nn.Linear(in_channels, hidden_channels, bias=False)
        self.right = nn.Linear(in_channels, hidden_channels, bias=False)
        self.linear_ff = nn.Linear(hidden_channels, in_channels, bias=False)

        self.dropout_module = nn.Dropout(dropout_p)
        self.act = comp_silu

    def forward(self, x):
        x = self.norm(x)

        return self.linear_ff(self.act(self.left(x)) * self.right(x))


class TriangularMultiplicative(nn.Module):
    def __init__(
            self,
            in_channels,
            scalen=1,
            dropout_p2d=0.1,
            split_seq=None,
    ):

        super(TriangularMultiplicative, self).__init__()

        hidden_channels = in_channels*scalen
        self.split_seq = split_seq

        self.norm1 = nn.RMSNorm(in_channels)
        self.linear_left = nn.Linear(in_channels, hidden_channels)
        self.linear_right = nn.Linear(in_channels, hidden_channels)
        self.gate_left = nn.Linear(in_channels, hidden_channels)
        self.gate_right = nn.Linear(in_channels, hidden_channels)

        self.gate_out = nn.Linear(in_channels, in_channels)

        self.norm2 = nn.RMSNorm(hidden_channels)
        self.linear_out = nn.Linear(hidden_channels, in_channels)

        self.dropout_module = nn.Dropout(p=dropout_p2d)

        self.act = comp_sigmoid

    def forward(self, pair, mode='outgoing'):

        pair = self.norm1(pair)

        left = self.act(self.gate_left(pair)) * self.linear_left(pair)
        right = self.act(self.gate_right(pair)) * self.linear_right(pair)
        gate = self.act(self.gate_out(pair))

        if mode == 'outgoing':
            if self.split_seq == None:
                out = gate * self.linear_out(self.norm2(comp_einsum('bilc, bjlc -> bijc', left, right)))
            else:
                full_out = []
                for x in torch.split(left, self.split_seq, dim=1):
                    x = self.linear_out(self.norm2(comp_einsum('bilc, bjlc -> bijc', x, right)))
                    full_out.append(x)
                out = gate*torch.concat(full_out, dim=1)

        else:
            if self.split_seq == None:
                out = gate * self.linear_out(self.norm2(comp_einsum('blic, bljc -> bijc', left, right)))
            else:
                full_out = []
                for x in torch.split(left, self.split_seq, dim=2):
                    x = self.linear_out(self.norm2(comp_einsum('blic, bljc -> bijc', x, right)))
                    full_out.append(x)
                out = gate*torch.concat(full_out, dim=1)

        return self.dropout_module(out)


class AxialFormer(nn.Module):

    def __init__(self,
                 in_channel,
                 nhead,
                 dropout_p=0.1,
                 tied_attn=False,
                 is_causal=False,
                 with_column_attn=True,
                 using_pwa_attn=False,
                 split_seq=None,
                 ):

        super(AxialFormer, self).__init__()

        self.in_channel = in_channel
        self.nhead = nhead
        self.with_column_attn = with_column_attn
        self.split_seq = split_seq

        self.row_attn = Attn(in_channels=self.in_channel,
                             nhead=self.nhead,
                             dropout_p=dropout_p,
                             is_causal=is_causal,
                             tied_attn=tied_attn)

        if self.with_column_attn:
            self.column_attn = Attn(in_channels=self.in_channel,
                                    nhead=self.nhead,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                    tied_attn=tied_attn)

        self.ffn = FeedForwardNetwork(in_channels=self.in_channel,
                                      dropout_p=dropout_p,
                                      )

    def forward(self, x, row_bias=None, column_bias=None):

        if self.split_seq is not None:
            full_out = []
            for x in torch.split(x, self.split_seq, dim=1):
                x += self.row_attn(x, row_bias)
                full_out.append(x)
            x = torch.concat(full_out, dim=1)
        else:
            x += self.row_attn(x, row_bias)

        if self.with_column_attn:
            x = x.transpose(1,2)
            if self.split_seq is not None:
                full_out = []
                for x in torch.split(x, self.split_seq, dim=1):
                    x += self.column_attn(x, column_bias)
                    full_out.append(x)
                x = torch.concat(full_out, dim=1)
            else:
                x += self.column_attn(x, column_bias)
            x = x.transpose(1,2)

        if self.split_seq is not None:
            full_out = []
            for x in torch.split(x, self.split_seq, dim=1):
                x += self.ffn(x)
                full_out.append(x)
            x = torch.concat(full_out, dim=1)
        else:
            x += self.ffn(x)
        return x


class Transpair(nn.Module):

    def __init__(self,
                 in_channel,
                 nhead,
                 tied_attn,
                 ):
        super(Transpair, self).__init__()

        self.tied_attn = tied_attn
        self.linear = nn.Linear(in_channel, nhead, bias=False)
        self.norm = nn.RMSNorm(in_channel)

    def forward(self, pair):

        pair = self.linear(self.norm(pair))
        pair = torch.permute(pair, [0, 3, 1, 2])

        if not self.tied_attn:
            pair = pair.unsqueeze(1)

        return pair


class TransMSA(nn.Module):

    def __init__(self,
                 msa_channel=128,
                 hidden_channel=32,
                 pair_channel=128,
                 dropout_p=0.1,
                 split_seq=None,
                 outnorm=False,
                 ):

        super(TransMSA, self).__init__()

        self.split_seq = split_seq

        self.norm = nn.RMSNorm(msa_channel)

        self.linear1 = nn.Linear(msa_channel, hidden_channel, bias=True)
        self.linear2 = nn.Linear(msa_channel, hidden_channel, bias=True)
        self.linear3 = nn.Linear(hidden_channel ** 2, pair_channel, bias=True)

        self.outnorm = outnorm
        if outnorm:
            self.norm2 = nn.RMSNorm(pair_channel)

        self.dropout_module = nn.Dropout2d(dropout_p)

    def _opm(self, a, b):

        batch_size, num_row, num_column, c = a.shape

        outer = comp_einsum("...mbc,...mde->...bdce", a, b) / num_row
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear3(outer)

        return outer

    def forward(self, msa):

        msa = self.norm(msa)

        left = self.linear1(msa)
        right = self.linear2(msa)

        if self.split_seq is not None:
            pair = []
            for x in torch.split(right, self.split_seq, dim=2):
                x = self._opm(left, x)
                pair.append(x)
            pair = torch.concat(pair, dim=2)
        else:
            pair = self._opm(left, right)

        if self.outnorm:
            pair = self.norm2(pair)

        return pair


class Evoformer(nn.Module):

    def __init__(self,
                 msa_channel,
                 msa_nhead,
                 pair_channel,
                 pair_nhead,
                 dropout_p=0.,
                 dropout_p2d=0.0,
                 with_column_attn=False,
                 using_pwa_attn=False,
                 tied_attn=False,
                 split_seq=None,
                 is_causal=False):

        super(Evoformer, self).__init__()

        self.split_seq = split_seq

        self.transmsa = TransMSA(msa_channel=msa_channel,
                                 pair_channel=pair_channel,
                                 dropout_p=dropout_p,
                                 split_seq=split_seq,
                                 outnorm=False)

        self.transpair = Transpair(in_channel=pair_channel,
                                   nhead=msa_nhead,
                                   tied_attn=tied_attn,
                                   )


        self.multi_outgoing = TriangularMultiplicative(in_channels=pair_channel,
                                                       dropout_p2d=dropout_p2d,
                                                       split_seq=split_seq)

        self.multi_incoming = TriangularMultiplicative(in_channels=pair_channel,
                                                       dropout_p2d=dropout_p2d,
                                                       split_seq=split_seq)

        self.triangle_startnode = Attn(in_channels=pair_channel,
                                       nhead=pair_nhead,
                                       dropout_p=dropout_p2d,
                                       is_causal=is_causal,
                                       tied_attn=tied_attn)

        self.triangle_endnode = Attn(in_channels=pair_channel,
                                     nhead=pair_nhead,
                                     dropout_p=dropout_p2d,
                                     is_causal=is_causal,
                                     tied_attn=tied_attn)

        self.bias_startnode = Transpair(in_channel=pair_channel,
                                        nhead=pair_nhead,
                                        tied_attn=tied_attn,
                                        )

        self.bias_endnode = Transpair(in_channel=pair_channel,
                                      nhead=pair_nhead,
                                      tied_attn=tied_attn,
                                      )

        self.pair_ffn = FeedForwardNetwork(in_channels=pair_channel,
                                           dropout_p=dropout_p2d,
                                           )


        self.axial = AxialFormer(in_channel=msa_channel,
                                 nhead=msa_nhead,
                                 dropout_p=dropout_p,
                                 tied_attn=tied_attn,
                                 is_causal=is_causal,
                                 with_column_attn=with_column_attn,
                                 using_pwa_attn=using_pwa_attn,
                                 split_seq=split_seq,
                                 )

    def forward(self, msa, pair, row_mask=None, column_mask=None):

        pair += self.transmsa(msa)

        pair += self.multi_outgoing(pair)
        pair += self.multi_incoming(pair, 'incoming')

        pair_bias = self.bias_startnode(pair)
        pair += self.triangle_startnode(pair, pair_bias)

        pair_bias = self.bias_endnode(pair).transpose(-2, -1)
        pair += self.triangle_endnode(pair.transpose(1,2), pair_bias).transpose(1,2)

        if self.split_seq is not None:
            full_out = []
            for x in torch.split(pair, self.split_seq, dim=1):
                x = x + self.pair_ffn(x)
                full_out.append(x)
            pair = torch.concat(full_out, dim=1)
        else:
            pair += self.pair_ffn(pair)

        # update msa part
        pair_bias = self.transpair(pair)
        msa = self.axial(msa, row_bias=pair_bias)

        return msa, pair


class InputEmbedder(nn.Module):
    def __init__(self,
                 pair_channel,
                 nums_aa=5,
                 atom_channel=256,
                 nums_posclass=65,
                 ):
        super(InputEmbedder, self).__init__()

        self.embd_pos = nn.Embedding(nums_posclass, pair_channel)
        self.embed_seq = nn.Embedding(nums_aa, atom_channel)

    def forward(self, monomer):

        idx = self.relpos(monomer['sel_idx']).to(monomer['na_coords'].device)
        seq_init = self.embed_seq(monomer['label_seq']).unsqueeze(1)

        return idx, self.embd_pos.weight, seq_init

    def relpos(self, idx, min_dis=-32, max_dis=32):

        idx = idx.unsqueeze(0) - idx.unsqueeze(1)
        idx = torch.clamp(idx, min_dis, max_dis) - min_dis

        return idx


class MSAEncoder(nn.Module):

    def __init__(self,
                 num_block,
                 msa_channel,
                 msa_nhead,
                 pair_channel,
                 pair_nhead,
                 tied_attn,
                 dropout_p,
                 dropout_p2d,
                 with_column_attn,
                 split_seq,
                 ):

        super(MSAEncoder, self).__init__()

        self.msa_channel = msa_channel
        self.msa_nhead = msa_nhead
        self.pair_channel = pair_channel
        self.pair_nhead = pair_nhead
        self.tied_attn = tied_attn
        self.dropout_p = dropout_p
        self.dropout_p2d = dropout_p2d
        self.with_column_attn = with_column_attn
        self.split_seq = split_seq

        self.msa_encoder = self._make_msa_encoder(num_block=num_block)

    def _make_msa_encoder(self, num_block):

        layers = []
        for index in range(num_block):
            layer = Evoformer(msa_channel=self.msa_channel,
                              msa_nhead=self.msa_nhead,
                              pair_channel=self.pair_channel,
                              pair_nhead=self.pair_nhead,
                              dropout_p=self.dropout_p,
                              dropout_p2d=self.dropout_p2d,
                              with_column_attn=self.with_column_attn,
                              tied_attn=self.tied_attn,
                              split_seq=self.split_seq,
                              is_causal=False)

            layers.append(('msa_block' + str(index), layer))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, msa, pair, msa_idx=None):

        for layer in self.msa_encoder:
            msa, pair = layer(msa, pair)

        return msa, pair


class ConfidenceHead(nn.Module):

    def __init__(self,
                 nblock,
                 atom_channel,
                 atom_nhead,
                 pair_channel,
                 pair_nhead,
                 dropout_p,
                 dropout_p2d,
                 split_seq,
                 no_bins_lddt=50,
                 min_dis=3.25,
                 max_dis=20.75,
                 no_bins_dis=16,
                 nums_aa=6,
                 tied_attn=False,
                 with_column_attn=False,
                 ):
        super(ConfidenceHead, self).__init__()

        self.min_dis = min_dis
        self.max_dis = max_dis
        self.no_bin_dis = no_bins_dis

        self.evoformer = MSAEncoder(num_block=nblock,
                                    msa_channel=atom_channel,
                                    msa_nhead=atom_nhead,
                                    pair_channel=pair_channel,
                                    pair_nhead=pair_nhead,
                                    tied_attn=tied_attn,
                                    dropout_p=dropout_p,
                                    dropout_p2d=dropout_p2d,
                                    with_column_attn=with_column_attn,
                                    split_seq=split_seq,
                                    )

        self.trans_dis = nn.Linear(no_bins_dis, pair_channel, bias=False)
        self.norm_pair = nn.RMSNorm(pair_channel)
        self.trans_pair = nn.Linear(pair_channel, pair_channel, bias=False)

        self.embed_seq = nn.Embedding(nums_aa, atom_channel)

        self.out_plddt = nn.Sequential(nn.Linear(atom_channel, no_bins_lddt, bias=False)) 

    def forward(self, seq, coords, pair_idx, pair_params, c3p_idx=5):

        seq = self.embed_seq(seq).unsqueeze(1)
        coords = coords.detach()[:, c3p_idx, ...]
        pair_params = self.trans_pair(self.norm_pair(pair_params.detach()))
        pair = torch.embedding(pair_params, pair_idx.int()).unsqueeze(0)

        bins = torch.linspace(self.min_dis, self.max_dis, self.no_bin_dis - 1, dtype=coords.dtype, device=coords.device, requires_grad=False, )
        bins = torch.cat([bins.new_tensor([0.0]), bins], dim=-1)
        upper = torch.cat([bins[1:], bins.new_tensor([1e10])], dim=-1)

        d = torch.cdist(coords, coords).unsqueeze(-1)
        d = ((d > bins) * (d < upper)).type(coords.dtype)

        pair = pair + self.trans_dis(d)

        for layer in self.evoformer.msa_encoder:
            seq, pair = layer(seq, pair)

        plddt = self.out_plddt(seq)

        return plddt


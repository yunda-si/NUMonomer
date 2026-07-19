#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yunda_si
"""

from ml_collections import config_dict


def get_cfg():
    config = config_dict.ConfigDict()

    config.model = config_dict.ConfigDict()
    config.model.blocks_structure_decoder = 8
    config.model.atom_channel = 384
    config.model.atom_nhead = 12
    config.model.pair_channel = 64
    config.model.pair_nhead = 4
    config.model.num_structure_recycle = 8
    config.model.num_atom = 28
    config.model.dropout_p = 0.15
    config.model.dropout_p2d = 0.15
    config.model.blocks_confidence = 4
    
    config.inference = config_dict.ConfigDict()
    config.inference.split_seq = None
    config.inference.split_atom = None
    config.inference.use_dsattn = False
    config.inference.disable_compile = True
    
    return config


































# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

import numpy as np


atom_types = ['P', 'OP1', 'OP2', "C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'",
              'N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', 'C2', 'C4', 'C5', 'C6', 'C8', 'O2', 'O4', 'O6', 'C7']
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 28.

element_types = ['N', 'O', 'P', 'C']
atom2element = np.zeros((atom_type_num, len(element_types)))
for atom_idx, atom in enumerate(atom_types):
    ele_idx = element_types.index(atom[:1])
    atom2element[atom_idx, ele_idx] = 1

residue_atoms = {
                "DA": ['P', 'OP1', 'OP2', "C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'",
                       'N1', 'N3', 'N6', 'N7', 'N9', 'C2', 'C4', 'C5', 'C6', 'C8'],
                "DT": ['P', 'OP1', 'OP2', "C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'",
                       'N1', 'N3', 'C2', 'C4', 'C5', 'C6', 'O2', 'O4', 'C7'],
                "DC": ['P', 'OP1', 'OP2', "C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'",
                       'N1', 'N3', 'N4', 'C2', 'C4', 'C5', 'C6', 'O2'],
                "DG": ['P', 'OP1', 'OP2', "C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'",
                       'N1', 'N2', 'N3', 'N7', 'N9', 'C2', 'C4', 'C5', 'C6', 'C8', 'O6'],
                "A":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
                       "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4" ],
                "U":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
                       "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
                "C":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
                       "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
                "G":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
                       "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
                "X":  ["C3'", 'C4', "O4'", "O5'", "C1'", "C4'", 'N3', 'C5', 'OP1', 'OP2',
                       'C2', 'P', "O3'", "C2'", 'N1', 'C6', "C5'"]
                }

van_der_waals_radius = {
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
    'P': 1.8,
}

restype_3to1 = {'DA': 'A',
                'DT': 'T',
                'DC': 'C',
                'DG': 'G',
                'A': 'A',
                'U': 'U',
                'C': 'C',
                'G': 'G',
                'X': 'X',
                }

restype_1to3 = {'dna':{'A':'DA',
                       'T':'DT',
                       'C':'DC',
                       'G':'DG',
                       'X':'X'},
                'rna':{'A':'A',
                       'U':'U',
                       'C':'C',
                       'G':'G',
                       'X':'X'}}

restypes1 = ['A', 'T', 'C', 'G', 'U']
restypes1_set = set(restypes1)
restype1_order = {restype: i for i, restype in enumerate(restypes1 + ['X'])}
restype_num1 = len(restype1_order)

restypes3 = [i for i in restype_3to1.keys()]
restypes3_set = set(restypes1)
restype3_order = {restype: i for i, restype in enumerate(restypes3 + ['X'])}


rev_restype1_order = {j:i for i,j in restype1_order.items()}
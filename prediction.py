# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

import random
from model import NUMonomer
import torch
from torch.utils.data import DataLoader
from dataset import MonomerDataset
import time
import warnings
warnings.filterwarnings('ignore')
from config import get_cfg
import json
from utils import save_struc, check_input
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from utils.seq_utils import compute_plddt
import argparse
import numpy as np

def cpu_analyze_and_store(atom_coords, bfactors, seq_list, idx_list, mol_type_list, chain_list, save_file,ftype):
    save_struc.to_pdb(atom_coords, bfactors, seq_list.squeeze(), idx_list, mol_type_list, chain_list, save_file,ftype)
    
def predicted(cfg, device, dataloader):

    #### model
    model = NUMonomer(blocks_structure_decoder=cfg.model.blocks_structure_decoder,
                      blocks_confidence=cfg.model.blocks_confidence,
                      atom_channel=cfg.model.atom_channel,
                      atom_nhead=cfg.model.atom_nhead,
                      pair_channel=cfg.model.pair_channel,
                      pair_nhead=cfg.model.pair_nhead,
                      dropout_p=cfg.model.dropout_p,
                      dropout_p2d=cfg.model.dropout_p2d,
                      num_atom=cfg.model.num_atom,
                      split_seq=cfg.inference.split_seq,
                      split_atom=cfg.inference.split_atom,
                      )
                  
    checkpoint = torch.load(cfg.weight_file, map_location='cpu')
    
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    weight_name = Path(weight_file).stem


    pending = []
    for d, total_feature in enumerate(dataloader):
        entry_path = os.path.join(cfg.save_path, total_feature['target_name'])
        if not os.path.exists(entry_path):
            os.mkdir(entry_path)
        json_file = os.path.join(entry_path, f'log_{weight_name}_{seed}.json')
        log_dict = {}

        for key, value in total_feature.items():
            if type(value) is torch.Tensor:
                total_feature[key] = value.to(device)

        torch.cuda.empty_cache()
        t1 = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            preds = model(total_feature,last_only=cfg.save_last)
        t2 = time.time()

        for key, value in total_feature.items():
            if type(value) == torch.Tensor:
                total_feature[key] = value.cpu()

        pred_coords = preds['pred_coords'].cpu()
        pred_lddts = torch.repeat_interleave(compute_plddt(preds['pred_plddts']).unsqueeze(-1), 28,-1).cpu()
        idx_maxplddt = max(torch.argmax(torch.mean(pred_lddts, dim=[1, 2, 3, 4])).item(), 3)
        if cfg.save_last:
            idx_coords = [len(pred_coords)-1]
        else:
            idx_coords = [idx_maxplddt]
        
        for idx_coord in idx_coords:
            save_file = os.path.join(entry_path, f'pred_{weight_name}_{seed}.cif')       

            f = cpu_pool.submit(cpu_analyze_and_store,
                             	pred_coords[idx_coord,0],
                             	pred_lddts[idx_coord, 0, ...,-1].squeeze(),
                             	total_feature['label_seq'],
                             	total_feature['sel_idx'],
                             	total_feature['mol_type'],
                             	total_feature['chain_list'],
                             	save_file,
                                cfg.ftype)
            pending.append(f)
            
        log_dict['timing'] = f'{t2-t1:6.2f}'
        log_dict['plddt'] = [f'{i.item():6.1f}' for i in torch.mean(pred_lddts, dim=[1, 2, 3, 4])]
        log_dict['len_seq'] = f"{len(total_feature['label_seq'].squeeze()):6d}"
        log_dict['target'] = total_feature['target_name']
        log_dict['idx_coords'] = idx_coords
        json.dump(log_dict, open(json_file,'w'))

        print(log_dict)
    
    for future in as_completed(pending):
        result = future.result()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NUMonomer')
    parser.add_argument('-seq_path', '--seq_path',
                        default=None,
                        type=str,
                        help="/mnt/example",
                        required=False)
    parser.add_argument('-seq_file', '--seq_file',
                        default=None,
                        type=str,
                        help="/mnt/example.fasta",
                        required=False)
    parser.add_argument('-save_path', '--save_path',
                        default=None,
                        type=str,
                        help='path to the save directory',
                        required=True)
    parser.add_argument('-weight', '--weight_file',
                        default=None,
                        type=str,
                        help='path to model weights',
                        required=True)
    parser.add_argument('-ftype', '--file_type',
                        default='cif',
                        type=str,
                        help='pdb or cif',
                        required=True)
    parser.add_argument('-device', '--device',
                        default='cuda:0',
                        type=str,
                        help='device to run the model',
                        required=False)
    parser.add_argument('-seed', '--random_seed',
                        default=42,
                        type=int,
                        help='random seed',
                        required=False)
    parser.add_argument('-last', '--save_last',
                        help='random seed',
                        action='store_true',
                        required=False)
    parser.add_argument('-ncpu', '--num_cpu',
                        default=8,
                        type=int,
                        help='threads',
                        required=False)
    parser.add_argument('-split_seq', '--split_seq',
                        default=0,
                        type=int,
                        help='split_seq',
                        required=False)
    parser.add_argument('-split_atom', '--split_atom',
                        default=0,
                        type=int,
                        help='split_atom',
                        required=False)
    parser.add_argument('-num_iter', '--num_iter',
                        default=8,
                        type=int,
                        help='num_iter',
                        required=False)
    parser.add_argument('-clamp_plddt', '--clamp_plddt',
                        default=512,
                        type=int,
                        help='split_seq',
                        required=False)
                        
    args = parser.parse_args()
    cfg = get_cfg()
    
    seq_path = args.seq_path
    seq_file = args.seq_file
    save_path = args.save_path
    weight_file = args.weight_file
    device = args.device
    ftype = args.file_type
    seed = args.random_seed
    save_last = args.save_last
    ncpu = args.num_cpu
    split_seq = args.split_seq
    split_atom = args.split_atom
    num_iter = args.num_iter
    clamp_plddt = args.clamp_plddt

    # init config
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    torch.set_num_threads(ncpu)
    cpu_pool = ProcessPoolExecutor(max_workers=ncpu)
    
    cfg.weight_file = weight_file
    cfg.save_path = save_path
    cfg.ftype = ftype
    cfg.save_last = save_last
    
    # inference
    if split_seq>0:
        cfg.inference.split_seq = split_seq
    if split_atom>0:
        cfg.inference.split_atom = split_atom

    # dataset
    if seq_path and seq_file:
        raise ValueError('Only seq_path or seq_file')

    if seq_file:
        targets_list=[seq_file]
    elif seq_path:
        targets_list=[os.path.join(seq_path,i) for i in os.listdir(seq_path)]
    else:
        raise ValueError('At least seq_path or seq_file')
        
    targets_list = check_input.check_input(targets_list)
    if len(targets_list)<1:
        raise ValueError('number of inputs==0')
    
    print('inputs:\n', '\n'.join(targets_list))
    
    dataset = MonomerDataset(targets_list=targets_list,
                             num_structure_recycle=num_iter,
                             clamp_plddt=clamp_plddt,
                             mask_seq=False,)

    dataloader = DataLoader(dataset, batch_size=None)
    if not os.path.exists(save_path):
        print(f'mkdir {save_path}')
        os.mkdir(save_path)
    
    predicted(cfg, device, dataloader)


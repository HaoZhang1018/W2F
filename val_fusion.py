import os
import sys
import argparse

import cv2
import time
import mmcv
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from datasets import FusionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config',
                        default="./config/MFNet_mit_b4_search_task.yaml",
                        help='train config file path')
    parser.add_argument('--data-dir',
                        default="./data",
                        help='test data root')
    parser.add_argument('--modal-x',
                        default="visible",
                        help='test data visible path')
    parser.add_argument('--modal-y',
                        default="infrared",
                        help='test data visible path')
    parser.add_argument(
        '--save-dir',
        default="./results_fusion/FMB",
        help='the dir to save logs and models')
    parser.add_argument(
        '--load-from',
        default="./checkpoints/MFNet/model.pth",
        help='the checkpoint file to resume from')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join('./results_fusion', os.path.basename(args.data_dir))

    mmcv.mkdir_or_exist(os.path.abspath(save_dir))

    test_dataset = FusionDataset(args.data_dir, args.modal_x, args.modal_y)
    print(f"Load {len(test_dataset)} file from {args.data_dir}")
    
    cfg = OmegaConf.load(args.config)
    if cfg.arch == "Task1":
        from models import SegTaskNet
        model = SegTaskNet(cfg, norm_layer=nn.BatchNorm2d)
    elif cfg.arch == "Task2":
        from models import FusionTaskNet
        model = FusionTaskNet(cfg, norm_layer=nn.BatchNorm2d)
    elif cfg.arch == "NAS_Task":
        from models import NDDRTaskNet
        model = NDDRTaskNet(cfg, norm_layer=nn.BatchNorm2d)
        
    # load checkpoint
    if args.load_from is not None:
        state_dict = torch.load(args.load_from, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(state_dict['model'], strict=True)
        except Exception as e:
            model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval()
    ndata = len(test_dataset)
    with torch.no_grad():
        for idx in tqdm(range(ndata)):
            modal_x, modal_y, image_size, filename = test_dataset[idx]

            modal_x = modal_x.to(device)
            modal_y = modal_y.to(device)

            results = model(modal_x, modal_y)
            out = results.out2[0]

            out = out.cpu().numpy().transpose([1, 2, 0])
            # recover from padding
            out = out[32:image_size[0]+32, 32:image_size[1]+32]
            
            out = np.array(out * 255, dtype=np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"{filename}.png"), out)
            

if __name__ == '__main__':
    main()
    

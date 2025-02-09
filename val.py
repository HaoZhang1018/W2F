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

from timm.models.layers import to_2tuple

from utils.loggers import get_root_logger
from utils.pyt_utils import ensure_dir
from utils.visualization import print_iou

from datasets.utils import normalize
from datasets.utils import pad_image_to_shape

from models.tasks import SegTask


class Evaluator(object):
    def __init__(self, cfg, dataset, model, task, device, save=False):
        self.num_classes = cfg.num_classes
        self.stride_rate = cfg.eval.eval_stride_rate
        self.multi_scales = cfg.eval.eval_scale_array
        self.is_flip = cfg.eval.eval_flip
        self.crop_size = cfg.eval.eval_crop_size
        
        self.dataset = dataset
        self.model = model
        self.task = task
        
        self.ndata = len(dataset)
        self.save_img = save
        self.device = device

        if self.save_img:
            self.save_path = cfg.save_dir

    def sliding_eval(self, modal_x, modal_y, crop_size=None, stride_rate=None):
        if crop_size is None:
            crop_size = self.crop_size
        if stride_rate is None:
            stride_rate = self.stride_rate
        
        crop_size = to_2tuple(crop_size)
        ori_rows, ori_cols, _ = modal_x.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.num_classes))

        for s in self.multi_scales:
            modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            modal_y_scale = cv2.resize(modal_y, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            processed_pred += self.scale_process(modal_x_scale, modal_y_scale,
                                                 (ori_rows, ori_cols),
                                                 crop_size, stride_rate)
        
        pred = processed_pred.argmax(2)
        return pred
    
    def scale_process(self, modal_x, modal_y, ori_shape, crop_size=None, stride_rate=None):
        new_rows, new_cols, c = modal_x.shape
        if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
            input_modal_x, input_modal_y, margin = self.process_image(modal_x, modal_y, crop_size)
            score = self.val_func_process(input_modal_x, input_modal_y)
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            input_modal_x, margin = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
            input_modal_y, margin = pad_image_to_shape(modal_y, crop_size, cv2.BORDER_CONSTANT, value=0)

            pad_rows, pad_cols = input_modal_x.shape[0], input_modal_x.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.num_classes, pad_rows, pad_cols).to(self.device)
            count_scale = torch.zeros(self.num_classes, pad_rows, pad_cols).to(self.device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    modal_x_sub = input_modal_x[s_y:e_y, s_x: e_x, :]
                    modal_y_sub = input_modal_y[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    modal_x_sub, modal_y_sub, tmargin = self.process_image(modal_x_sub, modal_y_sub, crop_size)
                    temp_score = self.val_func_process(modal_x_sub, modal_y_sub)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        
        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process(self, modal_x, modal_y):
        modal_x = np.ascontiguousarray(modal_x[None, :, :, :], dtype=np.float32)
        modal_x = torch.FloatTensor(modal_x).to(self.device)
        modal_y = np.ascontiguousarray(modal_y[None, :, :, :], dtype=np.float32)
        modal_y = torch.FloatTensor(modal_y).to(self.device)

        with torch.cuda.device(modal_x.get_device()):
            self.model.eval()
            with torch.no_grad():
                results = self.model(modal_x, modal_y)
                out = results.out1[0]

                if self.is_flip:
                    modal_x = modal_x.flip(-1)
                    modal_y = modal_y.flip(-1)
                    results_filp = self.model(modal_x, modal_y)
                    out_filp = results_filp.out1[0]
                    out += out_filp.flip(-1)
                
        return out

    def process_image(self, modal_x, modal_y, crop_size=None):
        modal_x, modal_y = normalize(modal_x), normalize(modal_y)
    
        if crop_size is not None:
            p_modal_x, margin = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_modal_y, _ = pad_image_to_shape(modal_y, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_modal_x = p_modal_x.transpose(2, 0, 1)
            p_modal_y = p_modal_y.transpose(2, 0, 1)
        
            return p_modal_x, p_modal_y, margin
    
        p_modal_x = p_modal_x.transpose(2, 0, 1) # 3 H W
        p_modal_y = p_modal_y.transpose(2, 0, 1)
    
        return p_modal_x, p_modal_x

    def evaluate(self, distributed=False):
        accumulator = []
        for idx in tqdm(range(self.ndata)):
            data = self.dataset[idx]

            modal_x = data['modal_x']
            modal_y = data['modal_y']
            label = data['label']
            name = data['fn']

            pred = self.sliding_eval(modal_x, modal_y)

            accumulator.append(
                self.task.accumulate_metric(pred, label)
                )
            
            if self.save_img:
                ensure_dir(self.save_path)
                fn = name + '.png'
                # save colored result
                class_colors = self.dataset.palette
                temp = np.zeros((pred.shape[0], pred.shape[1], 3))
                for i in range(self.num_classes):
                    temp[pred == i] = class_colors[i]

                cv2.imwrite(os.path.join(self.save_path, fn), temp)

        task_metric = self.task.aggregate_metric(accumulator)

        return task_metric
    

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config',
                        default="./config/MFNet_mit_b4_search_task.yaml",
                        help='train config file path')
    parser.add_argument(
        '--save-dir',
        default="./results/FMB",
        help='the dir to save logs and models')
    parser.add_argument(
        '--load-from',
        default="./checkpoints/MFNet/model.pth",
        help='the checkpoint file to resume from')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)

    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    elif cfg.get('save_dir', None) is None:
        cfg.save_dir = os.path.join('./results',
                                    os.path.splitext(os.path.basename(args.config))[0])
    
    mmcv.mkdir_or_exist(os.path.abspath(cfg.save_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.save_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    if "MFNet" in cfg.datasets.dataset_name:
        from datasets import MFNetDataset as RGBXDataset
    elif "FMB" in cfg.datasets.dataset_name:
        from datasets import FMBDataset as RGBXDataset
    
    test_dataset = RGBXDataset(cfg, stage="test")

    if cfg.arch == "NAS_Task":
        from models import NDDRTaskNet
        model = NDDRTaskNet(cfg, norm_layer=nn.BatchNorm2d)
    elif cfg.arch == "Task1":
        from models import SegTaskNet
        model = SegTaskNet(cfg, norm_layer=nn.BatchNorm2d)
    
    # load checkpoint
    if args.load_from is not None:
        state_dict = torch.load(args.load_from, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(state_dict['model'], strict=True)
        except Exception as e:
            model.load_state_dict(state_dict, strict=True)
        logger.info(args.load_from)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    task = SegTask(cfg.num_classes, cfg.weights.task1, cfg.datasets.ignore_index)

    eval = Evaluator(cfg, test_dataset, model, task, device, save=True)
    task_metric = eval.evaluate()
    print_str = print_iou(task_metric, class_names=test_dataset.classes)
    logger.info(print_str)


if __name__ == "__main__":
    main()


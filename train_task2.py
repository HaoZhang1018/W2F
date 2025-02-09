import os
import sys
import time
import random
import warnings
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import mmcv
from omegaconf import OmegaConf

from utils.loggers import get_root_logger
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.pyt_utils import all_reduce_tensor

from tensorboardX import SummaryWriter


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config',
                        default="./config/MFNet_mit_b4_nddr_task2.yaml",
                        help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from',
        help='the pretrained ckpt file to load from')
    parser.add_argument(
        '--resume-from',
        help='the checkpoint file to resume from')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    parser.add_argument("--port", type=int, default=29501)
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs',
                                    os.path.splitext(os.path.basename(args.config))[0])
        
    if args.load_from is not None:
            cfg.load_from = args.load_from
    elif cfg.get('load_from', None) is None:
        cfg.load_from = os.path.join('./pretrained', f"{cfg.backbone}.pth")
        
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    num_gpus = int(os.environ['WORLD_SIZE']) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    print(args.local_rank)
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", rank=args.local_rank
        )
        synchronize()
    
    if args.local_rank == 0:
        # create work_dir
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
        # dump config
        OmegaConf.save(cfg, os.path.join(cfg.work_dir, os.path.basename(args.config)), resolve=True)
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
        logger.info(cfg)

        # tb
        tb_dir =  os.path.join(cfg.work_dir, f'{timestamp}', "tb")
        writer = SummaryWriter(log_dir=tb_dir)

    meta = dict()
    # set random seeds
    if args.seed is not None:
        if args.local_rank == 0:
            logger.info(f'Set random seed to {args.seed}, deterministic: '
                        f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = os.path.basename(args.config)

    # set training model
    if distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    from models import FusionTaskNet
    model = FusionTaskNet(cfg, norm_layer=BatchNorm2d)
    
    if args.local_rank == 0:
        logger.info(f"{cfg.arch} model init done")

    # set dataloader
    from datasets import Train_pipline
    if cfg.datasets.dataset_name == "MFNetEnhance":
        from datasets import MFNetEnhanceDataset as RGBXDataset
    elif cfg.datasets.dataset_name == "FMBEnhance":
        from datasets import FMBEnhanceDataset as RGBXDataset

    train_process = Train_pipline(cfg)
    train_dataset = RGBXDataset(cfg, train_process, stage="train")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size if not distributed else cfg.train.batch_size // num_gpus,
        num_workers=cfg.train.num_workers,
        drop_last=True,
        shuffle=False if distributed else True,
        pin_memory=True,
        sampler=train_sampler
        )
    if args.local_rank == 0:
        logger.info(f'Load train datasets numbers: {len(train_dataset)}')

    # set optimizer
    base_lr = cfg.train.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=cfg.train.weight_decay)
    elif cfg.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    else:
        raise NotImplementedError
    
    # config lr policy
    niters_per_epoch = len(train_dataset) // cfg.train.batch_size # will drop last ones
    total_iteration = cfg.train.nepochs * niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, cfg.train.lr_power, total_iteration, niters_per_epoch * cfg.train.warm_up_epoch)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if distributed:
        model=torch.nn.DataParallel(model)
    
    # resume training
    state_epoch = 0
    if args.resume_from is not None:
        state_dict = torch.load(args.resume_from, map_location=torch.device('cpu'))
        state_epoch = state_dict['epoch'] + 1
        model.load_state_dict(state_dict['model'], strict=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        if args.local_rank == 0:
            logger.info(f"Resume checkpoint from {args.resume_from}")

    optimizer.zero_grad()
    model.train()
    if args.local_rank == 0:
        logger.info('begin trainning:')
    for epoch in range(state_epoch, cfg.train.nepochs+1):
        if distributed:
            train_sampler.set_epoch(epoch)  # steps is used to seed RNG
        
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            minibatch = dataloader.next()
            modal_x = minibatch['modal_x']
            modal_y = minibatch['modal_y']
            label = minibatch['label']
            label_x = minibatch['label_x']
            label_y = minibatch['label_y']
            Mask = minibatch['Mask']

            modal_x = modal_x.cuda(non_blocking=True)
            modal_y = modal_y.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            label_x = label_x.cuda(non_blocking=True)
            label_y = label_y.cuda(non_blocking=True)
            Mask = Mask.cuda(non_blocking=True)

            if distributed:
                results = model.module.loss(modal_x, modal_y, label_x, label_y, Mask, label)
            else:
                results = model.loss(modal_x, modal_y, label_x, label_y, Mask, label)
            
            loss = results.loss

            if distributed:
                loss = all_reduce_tensor(loss, world_size=num_gpus)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = epoch * niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            
            sum_loss += loss
            print_str = 'Epoch {}/{}'.format(epoch, cfg.train.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
            pbar.set_description(print_str, refresh=False)
        
        if (distributed and (args.local_rank == 0)) or (not distributed):
            writer.add_scalar('train_loss', sum_loss / len(pbar), epoch)
    
            latest_epoch_checkpoint = os.path.join(cfg.work_dir, f'latest.pth')
            state_dict = model.module.state_dict() if distributed else model.state_dict()
            checkpoint = {
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, latest_epoch_checkpoint)

            if (epoch >= cfg.checkpoint.start_epoch) and (epoch % cfg.checkpoint.step == 0) or (epoch == cfg.train.nepochs):
                current_epoch_checkpoint = os.path.join(cfg.work_dir, f'epoch-{epoch}.pth')
                logger.info("Saving checkpoint to file {}".format(current_epoch_checkpoint))
                torch.save(checkpoint, current_epoch_checkpoint)

            del state_dict, checkpoint
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
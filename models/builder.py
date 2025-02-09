import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import SegTask, FusionTask

from .encoders.dual_segformer import RGBXTransformerBlock
from .common_layers import get_nddr, NDDR_Brige
from .decoders.MLPDecoder import DecoderHead
from .decoders.CNNDecoder import CNNHead

from utils import AttrDict


class SegTaskNet(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(SegTaskNet, self).__init__()
        ## task auxiliary training
        self.task = SegTask(cfg.num_classes, cfg.weights.task1, cfg.datasets.ignore_index)

        ## params
        self.cfg = cfg
        ## define network
        self.num_classes =  cfg.num_classes
        self.in_chans = cfg.in_chans
        # encoder
        self.num_stages = cfg.num_stages
        img_size = cfg.encoder.img_size
        patch_size = cfg.encoder.patch_size
        stride = cfg.encoder.stride
        depths = cfg.encoder.depths
        embed_dims = cfg.encoder.embed_dims
        num_heads = cfg.encoder.num_heads
        mlp_ratios = cfg.encoder.mlp_ratios
        sr_ratios = cfg.encoder.sr_ratios
        
        dpr = [x.item() for x in torch.linspace(0, cfg.encoder.drop_path_rate, sum(depths))]

        cur = 0
        encoders = []
        for stage_id in range(self.num_stages):
            encoder = RGBXTransformerBlock(img_size=img_size, patch_size=patch_size[stage_id], stride=stride[stage_id], depths=depths[stage_id],
                                           in_chans=self.in_chans if stage_id == 0 else embed_dims[stage_id-1], embed_dims=embed_dims[stage_id], 
                                           num_heads=num_heads[stage_id], mlp_ratios=mlp_ratios[stage_id], qkv_bias=cfg.encoder.qkv_bias,
                                           qk_scale=cfg.encoder.qk_scale, drop_rate=cfg.encoder.drop_rate, attn_drop_rate=cfg.encoder.attn_drop_rate,
                                           drop_path_rate=dpr[cur:cur+depths[stage_id]], sr_ratios=sr_ratios[stage_id],
                                           after_norm=norm_layer, use_after_norm=False)
            cur += depths[stage_id]
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        self.decoder = DecoderHead(in_channels=cfg.decoder.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder.embed_dim)

        self._step = 0
        # load from
        self.init_weights(cfg.get('load_from', None))

    def step(self):
        self._step += 1

    def init_weights(self, model_file):
        if model_file is None:
            return
        if isinstance(model_file, str):
            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            state_dict = model_file

        self.load_state_dict(state_dict, strict=False)
        del state_dict

    def loss(self, modal_x, modal_y, label):
        result = self.forward(modal_x, modal_y)
        result.loss = self.task.loss(result.out1, label)
        return result

    def forward(self, modal_x, modal_y):
        out_semantic = []
        for stage_id in range(self.num_stages):
            modal_x, modal_y = self.encoders[stage_id].forward(modal_x, modal_y)
            fuse_xy = self.encoders[stage_id].forward_features(modal_x, modal_y)
            out_semantic.append(fuse_xy)

        out = self.decoder.forward(out_semantic)

        return AttrDict({'out1': out, 'out2': None})


class FusionTaskNet(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(FusionTaskNet, self).__init__()
        ## task auxiliary training
        self.task = FusionTask(cfg.weights.task2, cfg.loss.method, cfg.loss.patch_size)
        ## params
        self.cfg = cfg
        ## define network
        self.in_chans = cfg.in_chans
        # encoder
        self.num_stages = cfg.num_stages
        img_size = cfg.encoder.img_size
        patch_size = cfg.encoder.patch_size
        stride = cfg.encoder.stride
        depths = cfg.encoder.depths
        embed_dims = cfg.encoder.embed_dims
        num_heads = cfg.encoder.num_heads
        mlp_ratios = cfg.encoder.mlp_ratios
        sr_ratios = cfg.encoder.sr_ratios
        
        dpr = [x.item() for x in torch.linspace(0, cfg.encoder.drop_path_rate, sum(depths))]

        cur = 0
        encoders = []
        for stage_id in range(self.num_stages):
            encoder = RGBXTransformerBlock(img_size=img_size, patch_size=patch_size[stage_id], stride=stride[stage_id], depths=depths[stage_id],
                                           in_chans=self.in_chans if stage_id == 0 else embed_dims[stage_id-1], embed_dims=embed_dims[stage_id], 
                                           num_heads=num_heads[stage_id], mlp_ratios=mlp_ratios[stage_id], qkv_bias=cfg.encoder.qkv_bias,
                                           qk_scale=cfg.encoder.qk_scale, drop_rate=cfg.encoder.drop_rate, attn_drop_rate=cfg.encoder.attn_drop_rate,
                                           drop_path_rate=dpr[cur:cur+depths[stage_id]], sr_ratios=sr_ratios[stage_id],
                                           after_norm=norm_layer, use_after_norm=True)
            cur += depths[stage_id]
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        # decoder
        self.decoder = CNNHead(in_channels=cfg.decoder.channels, norm_layer=norm_layer)
        self._step = 0
        # load from
        self.init_weights(cfg.get('load_from', None))

    def step(self):
        self._step += 1

    def init_weights(self, model_file):
        if model_file is None:
            return
        if isinstance(model_file, str):
            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            state_dict = model_file

        self.load_state_dict(state_dict, strict=False)
        del state_dict

    def loss(self, modal_x, modal_y, label_x, label_y, Mask, label):
        result = self.forward(modal_x, modal_y)
        result.loss = self.task.loss(result.out2, label_x, label_y, Mask, label)
        return result

    def forward(self, modal_x, modal_y):
        out_visual = [torch.cat([modal_x, modal_y], dim=1)]
        for stage_id in range(self.num_stages):
            modal_x, modal_y = self.encoders[stage_id].forward(modal_x, modal_y)
            # fuse_xy = torch.cat([modal_x, modal_y], dim=1)
            fuse_xy = self.encoders[stage_id].forward_features(modal_x, modal_y)
            out_visual.append(fuse_xy)

        out = self.decoder.forward(out_visual)

        return AttrDict({'out1': None, 'out2': out})


class NDDRTaskNet(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(NDDRTaskNet, self).__init__()
        ## task auxiliary training
        self.task1 = SegTaskNet(cfg, norm_layer)
        self.task2 = FusionTaskNet(cfg, norm_layer)
        # load from
        self.load_weights(cfg.get('load_task1', None), cfg.get('load_task2', None))

        ## params
        self.cfg = cfg
        ## define network
        self.num_classes =  cfg.num_classes
        self.in_chans = cfg.in_chans
        # encoder
        self.num_stages = cfg.num_stages
        embed_dims = cfg.encoder.embed_dims

        # distillation brige
        # briges = []
        # for stage_id in range(self.num_stages):
        #     out_channels = embed_dims[stage_id]
        #     brige = NDDR_Brige(cfg, out_channels)
        #     briges.append(brige)
        # self.task1.brige = nn.ModuleList(briges)
        # self.task2.brige = nn.ModuleList(briges)

        # nddr
        nddrs = []
        for stage_id in range(self.num_stages):
            out_channels = embed_dims[stage_id]
            nddr = get_nddr(cfg, out_channels, out_channels, cfg.encoder.num_heads[stage_id], cfg.encoder.sr_ratios[stage_id], norm_layer=norm_layer)
            nddrs.append(nddr)
        self.nddrs = nn.ModuleList(nddrs)
        
        self._step = 0

    def step(self):
        self._step += 1

    def train_parameters(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return params

    def load_weights(self, task_file1, task_file2):
        if task_file1 is None or task_file2 is None:
            return
        if isinstance(task_file1, str):
            state_dict = torch.load(task_file1, map_location=torch.device('cpu'))
        else:
            state_dict = task_file1
        self.task1.load_state_dict(state_dict, strict=False)

        # freeze encoders except attention block
        for name, param in self.task1.encoders.named_parameters():
            param.requires_grad = False

        if isinstance(task_file2, str):
            state_dict = torch.load(task_file2, map_location=torch.device('cpu'))
        else:
            state_dict = task_file2
        self.task2.load_state_dict(state_dict, strict=False)
        
        # freeze encoders except attention block
        for name, param in self.task2.encoders.named_parameters():
            param.requires_grad = False

        del state_dict

    def loss(self, modal_x, modal_y, label, label_x, label_y, Mask):
        result = AttrDict({})

        out_semantic = []
        out_visual = [torch.cat([modal_x, modal_y], dim=1)]

        feat_x_sem, feat_y_sem = modal_x, modal_y
        feat_x_vis, feat_y_vis = modal_x, modal_y
        for stage_id in range(self.num_stages):
            feat_x_sem, feat_y_sem = self.task1.encoders[stage_id].forward(feat_x_sem, feat_y_sem)
            fuse_xy_sem = self.task1.encoders[stage_id].forward_features(feat_x_sem, feat_y_sem)

            feat_x_vis, feat_y_vis = self.task2.encoders[stage_id].forward(feat_x_vis, feat_y_vis)
            fuse_xy_vis = self.task2.encoders[stage_id].forward_features(feat_x_vis, feat_y_vis)

            # feature search
            search_feat_sem, search_feat_vis = self.nddrs[stage_id](fuse_xy_sem, fuse_xy_vis)
            # fuse_xy_sem = self.task1.brige[stage_id](fuse_xy_sem)
            # fuse_xy_vis = self.task2.brige[stage_id](fuse_xy_vis)

            out_semantic.append(search_feat_sem)
            out_visual.append(search_feat_vis)

        result.out1 = self.task1.decoder.forward(out_semantic)
        result.out2 = self.task2.decoder.forward(out_visual)

        result.loss1 = self.task1.task.loss(result.out1, label)
        result.loss2 = self.task2.task.loss(result.out2, label_x, label_y, Mask, label)
        result.loss = result.loss1 + self.cfg.weights.FACTOR * result.loss2

        return result
    

    def forward(self, modal_x, modal_y):
        out_semantic = []
        out_visual = [torch.cat([modal_x, modal_y], dim=1)]

        feat_x_sem, feat_y_sem = modal_x, modal_y
        feat_x_vis, feat_y_vis = modal_x, modal_y
        for stage_id in range(self.num_stages):
            feat_x_sem, feat_y_sem = self.task1.encoders[stage_id].forward(feat_x_sem, feat_y_sem)
            fuse_xy_sem = self.task1.encoders[stage_id].forward_features(feat_x_sem, feat_y_sem)

            feat_x_vis, feat_y_vis = self.task2.encoders[stage_id].forward(feat_x_vis, feat_y_vis)
            fuse_xy_vis = self.task2.encoders[stage_id].forward_features(feat_x_vis, feat_y_vis)

            search_feat_sem, search_feat_vis = self.nddrs[stage_id](fuse_xy_sem, fuse_xy_vis)
            # fuse_xy_sem = self.task1.brige[stage_id](fuse_xy_sem)
            # fuse_xy_vis = self.task2.brige[stage_id](fuse_xy_vis)

            out_semantic.append(search_feat_sem)
            out_visual.append(search_feat_vis)

        out1 = self.task1.decoder.forward(out_semantic)
        out2 = self.task2.decoder.forward(out_visual)

        return AttrDict({'out1': out1, 'out2': out2})


class SegTaskNDDRNet(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(SegTaskNDDRNet, self).__init__()
        ## task auxiliary training
        self.task1 = SegTaskNet(cfg, norm_layer)
        self.task2 = FusionTaskNet(cfg, norm_layer)
        # load from
        self.load_weights(cfg.get('load_task1', None), cfg.get('load_task2', None))

        ## params
        self.cfg = cfg
        ## define network
        self.num_classes =  cfg.num_classes
        self.in_chans = cfg.in_chans
        # encoder
        self.num_stages = cfg.num_stages
        embed_dims = cfg.encoder.embed_dims

        # nddr
        nddrs = []
        for stage_id in range(self.num_stages):
            out_channels = embed_dims[stage_id]
            nddr = get_nddr(cfg, out_channels, out_channels)
            nddrs.append(nddr)
        self.nddrs = nn.ModuleList(nddrs)
        
        self._step = 0

    def step(self):
        self._step += 1

    def train_parameters(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return params

    def load_weights(self, task_file1, task_file2):
        if task_file1 is None or task_file2 is None:
            return
        if isinstance(task_file1, str):
            state_dict = torch.load(task_file1, map_location=torch.device('cpu'))
        else:
            state_dict = task_file1
        self.task1.load_state_dict(state_dict, strict=False)

        # freeze encoders
        for param in self.task1.encoders.parameters():
            param.requires_grad = False

        if isinstance(task_file2, str):
            state_dict = torch.load(task_file2, map_location=torch.device('cpu'))
        else:
            state_dict = task_file2
        self.task2.load_state_dict(state_dict, strict=False)
        
        # freeze task2 params
        for param in self.task2.parameters():
            param.requires_grad = False

        del state_dict

    def loss(self, modal_x, modal_y, label):
        result = AttrDict({})

        out_semantic = []

        feat_x_sem, feat_y_sem = modal_x, modal_y
        feat_x_vis, feat_y_vis = modal_x, modal_y
        for stage_id in range(self.num_stages):
            feat_x_sem, feat_y_sem = self.task1.encoders[stage_id].forward(feat_x_sem, feat_y_sem)
            fuse_xy_sem = self.task1.encoders[stage_id].forward_features(feat_x_sem, feat_y_sem)

            feat_x_vis, feat_y_vis = self.task2.encoders[stage_id].forward(feat_x_vis, feat_y_vis)
            fuse_xy_vis = self.task2.encoders[stage_id].forward_features(feat_x_vis, feat_y_vis)

            # feature search
            search_feat_sem, _ = self.nddrs[stage_id](fuse_xy_sem, fuse_xy_vis)

            out_semantic.append(search_feat_sem)

        result.out1 = self.task1.decoder.forward(out_semantic)

        result.loss = self.task1.task.loss(result.out1, label)

        return result
    

    def forward(self, modal_x, modal_y):
        out_semantic = []

        feat_x_sem, feat_y_sem = modal_x, modal_y
        feat_x_vis, feat_y_vis = modal_x, modal_y
        for stage_id in range(self.num_stages):
            feat_x_sem, feat_y_sem = self.task1.encoders[stage_id].forward(feat_x_sem, feat_y_sem)
            fuse_xy_sem = self.task1.encoders[stage_id].forward_features(feat_x_sem, feat_y_sem)

            feat_x_vis, feat_y_vis = self.task2.encoders[stage_id].forward(feat_x_vis, feat_y_vis)
            fuse_xy_vis = self.task2.encoders[stage_id].forward_features(feat_x_vis, feat_y_vis)

            search_feat_sem, _ = self.nddrs[stage_id](fuse_xy_sem, fuse_xy_vis)

            out_semantic.append(search_feat_sem)

        out1 = self.task1.decoder.forward(out_semantic)

        return AttrDict({'out1': out1, 'out2': None})
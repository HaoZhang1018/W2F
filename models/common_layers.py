import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from timm.models.layers import trunc_normal_
from .net_utils import SelfAttention
    
def batch_norm(norm_layer=nn.BatchNorm2d, num_features=64, eps=1e-3, momentum=0.05):
    bn = norm_layer(num_features, eps, momentum)
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)
    return bn


def get_nddr_bn(cfg, norm_layer=nn.BatchNorm2d):
    if cfg.nddr.NDDR_BN_TYPE == 'default':
        return lambda width: batch_norm(norm_layer, width, eps=1e-3, momentum=cfg.nddr.BATCH_NORM_MOMENTUM)
    else:
        raise NotImplementedError
        

def get_nddr(cfg, in_channels, out_channels, nums_head, sr_ratio, norm_layer=nn.BatchNorm2d):
    if cfg.nddr.SEARCHSPACE == '':
        assert in_channels == out_channels
        if cfg.nddr.NDDR_TYPE == '':
            return NDDR(cfg, out_channels, norm_layer=norm_layer)
        elif cfg.nddr.NDDR_TYPE == 'group_attention':
            return AttentionSearch(cfg, out_channels, nums_head, sr_ratio, norm_layer=norm_layer)
        elif cfg.nddr.NDDR_TYPE == 'single_side':
            return SingleSideNDDR(cfg, out_channels, False)
        elif cfg.nddr.NDDR_TYPE == 'single_side_reverse':
            return SingleSideNDDR(cfg, out_channels, True)
        elif cfg.nddr.NDDR_TYPE == 'cross_stitch':
            return CrossStitch(cfg, out_channels)
        else:
            raise NotImplementedError
    elif cfg.nddr.SEARCHSPACE == 'GeneralizedMTLNAS':
        if cfg.nddr.NDDR_TYPE == '':
            return SingleSidedAsymmetricNDDR(cfg, in_channels, out_channels)
        elif cfg.nddr.NDDR_TYPE == 'cross_stitch':
            return SingleSidedAsymmetricCrossStitch(cfg, in_channels, out_channels)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


class CrossStitch(nn.Module):
    def __init__(self, cfg, out_channels):
        super(CrossStitch, self).__init__()
        init_weights = cfg.nddr.INIT
        
        self.a11 = nn.Parameter(torch.tensor(init_weights[0]))
        self.a22 = nn.Parameter(torch.tensor(init_weights[0]))
        self.a12 = nn.Parameter(torch.tensor(init_weights[1]))
        self.a21 = nn.Parameter(torch.tensor(init_weights[1]))

    def forward(self, feature1, feature2):
        out1 = self.a11 * feature1 + self.a21 * feature2
        out2 = self.a12 * feature1 + self.a22 * feature2
        return out1, out2
    
    
class NDDR(nn.Module):
    def __init__(self, cfg, out_channels, norm_layer=nn.BatchNorm2d):
        super(NDDR, self).__init__()
        init_weights = cfg.nddr.INIT
        norm = get_nddr_bn(cfg, norm_layer=norm_layer)
        
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        
        # Initialize weight
        if len(init_weights):
            self.conv1.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
            self.conv2.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[1],
                torch.eye(out_channels) * init_weights[0]
            ], dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.activation = nn.ReLU()

        self.bn1 = norm(out_channels)
        self.bn2 = norm(out_channels)

    def forward(self, feature1, feature2):
        x = torch.cat([feature1, feature2], 1)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out1 = self.bn1(out1)
        out2 = self.bn2(out2)
        out1 = self.activation(out1)
        out2 = self.activation(out2)
        return out1, out2


class CrossNDDR(nn.Module):
    def __init__(self, cfg, out_channels, norm_layer=nn.BatchNorm2d):
        super(CrossNDDR, self).__init__()
        init_weights = cfg.nddr.INIT
        norm = get_nddr_bn(cfg, norm_layer=norm_layer)

        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)

        # fusion conv
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.fusion_bn = norm(out_channels)

        self.bn1 = norm(out_channels)
        self.bn2 = norm(out_channels)

        self.activation = nn.ReLU()

        self._init_weights(out_channels, init_weights)

    def _init_weights(self, out_channels, init_weights=None):
        # Initialize weight
        if len(init_weights):
            self.fusion_conv.weight = nn.Parameter(torch.eye(out_channels * 2).view(out_channels, -1, 1, 1))
            self.conv1.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
            self.conv2.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[1],
                torch.eye(out_channels) * init_weights[0]
            ], dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.fusion_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, feature1, feature2):
        out = self.fusion_conv(torch.cat([feature1, feature2], 1))
        out = self.fusion_bn(out)

        out1 = self.conv1(out1)
        out2 = self.conv2(out2)
        out1 = self.bn1(out1)
        out2 = self.bn2(out2)

        out1 = self.activation(out1)
        out2 = self.activation(out2)

        return out1, out2


class AttentionSearch(nn.Module):
    def __init__(self, cfg, out_channels, num_heads=1, sr_ratio=1., attn_drop=0.0, proj_drop=0.0, norm_layer=nn.BatchNorm2d):
        super(AttentionSearch, self).__init__()
        init_weights = cfg.nddr.INIT
        norm = get_nddr_bn(cfg, norm_layer=norm_layer)

        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)

        self.bn1 = norm(out_channels)
        self.bn2 = norm(out_channels)

        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.fusion_bn = norm(out_channels)

        self.SA = SelfAttention(out_channels, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)

        self.activation = nn.ReLU()

        self._init_weights(out_channels, init_weights)

    def _init_weights(self, out_channels, init_weights=None):
        # Initialize weight
        if len(init_weights):
            self.fusion_conv.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * 0.5,
                torch.eye(out_channels) * 0.5
            ], dim=1).view(out_channels, -1, 1, 1))
            self.conv1.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
            self.conv2.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.fusion_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, feature1, feature2):
        B1, C1, H1, W1 = feature1.shape
        B2, C2, H2, W2 = feature2.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "feature1 and feature2 should have the same dimensions"
 
        out = self.fusion_conv(torch.cat([feature1, feature2], 1))
        out = self.fusion_bn(out)

        out_flat = out.flatten(2).transpose(1, 2)  ##B HXW C

        out = self.SA(out_flat, H1, W1)  ##B HXW C
        out = out.permute(0, 2, 1).reshape(B1, C1, H1, W1).contiguous()

        out1 = self.conv1(torch.cat([feature1, out], 1))
        out2 = self.conv2(torch.cat([feature2, out], 1))
        out1 = self.bn1(out1)
        out2 = self.bn2(out2)
        out1 = self.activation(out1)
        out2 = self.activation(out2)

        return out1, out2


class SingleSideNDDR(nn.Module):
    def __init__(self, cfg, out_channels, reverse):
        """
        Net1 is main task, net2 is aux task
        """
        super(SingleSideNDDR, self).__init__()
        init_weights = cfg.nddr.INIT
        norm = get_nddr_bn(cfg)

        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)

        # Initialize weight
        if len(init_weights):
            self.conv.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.activation = nn.ReLU()

        self.bn = norm(out_channels)

        self.reverse = reverse

    def forward(self, feature1, feature2):
        if self.reverse:
            out2 = feature2
            out1 = torch.cat([feature1, feature2], 1)
            out1 = self.conv(out1)
            out1 = self.bn(out1)
        else:
            out1 = feature1
            out2 = torch.cat([feature2, feature1], 1)
            out2 = self.conv(out2)
            out2 = self.bn(out2)
        return out1, out2


class SingleSidedAsymmetricCrossStitch(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super(SingleSidedAsymmetricCrossStitch, self).__init__()
        init_weights = cfg.nddr.INIT
        
        assert in_channels >= out_channels
        # check if out_channel divides in_channels
        assert in_channels % out_channels == 0
        multipiler = in_channels / out_channels - 1
        self.a = nn.Parameter(torch.tensor([init_weights[0]] +\
                                            [init_weights[1] / float(multipiler) for _ in range(int(multipiler))]))

    def forward(self, features):
        """

        :param features: upstream feature maps
        :return:
        """
        out = 0.
        for i, feature in enumerate(features):
            out += self.a[i] * feature
        return out
    
    
class SingleSidedAsymmetricNDDR(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super(SingleSidedAsymmetricNDDR, self).__init__()
        init_weights = cfg.nddr.INIT
        norm = get_nddr_bn(cfg)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        assert in_channels >= out_channels
        # check if out_channel divides in_channels
        assert in_channels % out_channels == 0
        multipiler = in_channels / out_channels - 1
        
        # Initialize weight
        if len(init_weights):
            weight = [torch.eye(out_channels) * init_weights[0]] +\
                 [torch.eye(out_channels) * init_weights[1] / float(multipiler) for _ in range(int(multipiler))]
            self.conv.weight = nn.Parameter(torch.cat(weight, dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.activation = nn.ReLU()
        self.bn = norm(out_channels)
        nn.init.constant_(self.bn.weight, 1.)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, features):
        """

        :param features: upstream feature maps
        :return:
        """
        x = torch.cat(features, 1)
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out

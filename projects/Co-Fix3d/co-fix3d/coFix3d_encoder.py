# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import torch
from mmcv.cnn import build_conv_layer
from torch import nn
from .encoder_utils import *
from .time_utils import T
import torchvision.models.resnet as resnet
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule, build_conv_layer
from einops import rearrange
import matplotlib.pyplot as plt
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from .attention import MultiheadFlashAttention
from .ops.msmv_sampling.wrapper import msmv_sampling
from functools import partial
# from timm.models.layers import trunc_normal_, DropPath
# from .decoder_utils  import M

@MODELS.register_module()
class CoFix3DEncoder(nn.Module):
    def __init__(self,
                 hidden_channel=128,
                 in_channels_pts=128 * 3,
                 bn_momentum=0.1,
                 step=1,
                 input_img=True,
                 input_pts=True,
                 cam_lss=False,
                 newbevpool=True,
                 pc_range=None,
                 img_scale=None,
                 bias='auto',
                 ):
        super(CoFix3DEncoder, self).__init__()

        self.input_pts = input_pts
        self.input_img = input_img
        self.cam_proj_type = cam_lss
        self.step = step
        if self.input_pts:
            self.shared_conv_pts = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_pts,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,

            )
        if self.input_img:

            from .lss import LiftSplatShoot
            self.cam_lss = LiftSplatShoot(grid=0.6, inputC=256, outputC=hidden_channel, camC=64,
                                          pc_range=pc_range, img_scale=img_scale, downsample=4, newbevpool=newbevpool)
            # self.in_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=None)
            # self.out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=None)
            # self.c_img_bev = ConstructImgBEV(hidden_channel=256)
        self.bn_momentum = bn_momentum
        if self.input_pts and self.input_img:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(512+hidden_channel, hidden_channel, 3, padding=1),
                nn.BatchNorm2d(hidden_channel))
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, pts_feats, img_feats, img_metas):
        batch_size = len(img_metas)
        # new_pts_feat = self.shared_conv_pts(pts_feats[0])
        if self.input_pts:
            new_pts_feat = self.shared_conv_pts(pts_feats[0])
            if self.step == 1:
                return new_pts_feat, None

        # img_feats=self.c_img_bev(new_pts_feat, img_feats, img_metas)

        if self.input_img :
            img_feats = img_feats[0]
            if self.cam_proj_type == 'proj':
                img_feats = self.cam_lss(img_feats.view(batch_size, -1, *img_feats.shape[-3:]), img_metas=img_metas)
            else:
                mat = img_metas[0]['lidar2img']
                inverse_mat = mat.inverse()
                rots = inverse_mat[..., :3, :3]
                trans = inverse_mat[..., :3, 3]
                img_feats, depth = self.cam_lss(img_feats.view(batch_size, -1, *img_feats.shape[-3:]), rots=rots, trans=trans, img_metas=img_metas)
                if self.step == 2:
                    return img_feats, None
                # gt_depth = img_metas['gt_depth']
                # loss_depth = self.cam_lss.get_depth_loss(gt_depth, depth)
                # new_pts_feat = self.c_img_bev(pts_feats[0], img_feats, img_metas)
        # if self.input_img and not self.input_pts:
        #     return None, img_feats
        if self.step==3:
            new_pts_feat = torch.cat([pts_feats[0], img_feats], dim=1)
            new_pts_feat = self.reduce_conv(new_pts_feat)
            # new_pts_feat = self.c_img_bev(pts_feats[0], img_feats,img_metas)
            return new_pts_feat, img_feats


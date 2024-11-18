import math
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
from mmcv.cnn import ConvModule
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


@MODELS.register_module()
class CoFix3D(MVXTwoStageDetector):

    def __init__(self, data_preprocessor, stop_prev_grad=0, freeze_pts=False, freeze_img=False, input_img=False, input_pts=False, **kwargs):
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        self.freeze_pts = freeze_pts
        self.freeze_img = freeze_img
        self.input_img = input_img
        self.input_pts = input_pts
        self.stop_prev_grad = stop_prev_grad
        super(BEVFormerFusion, self).__init__(data_preprocessor=data_preprocessor, **kwargs)
        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
            self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if hasattr(self, "img_backbone"):
            self.img_backbone.init_weights()

        if self.freeze_pts:
            for name, param in self.named_parameters():
                if 'pts' in name and 'pts_bbox_head' not in name and 'pts_fusion_layer' not in name:
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

            self.pts_voxel_layer.apply(fix_bn)
            self.pts_voxel_encoder.apply(fix_bn)
            self.pts_middle_encoder.apply(fix_bn)
            self.pts_backbone.apply(fix_bn)
            self.pts_neck.apply(fix_bn)

        if self.freeze_img:
            for name, param in self.named_parameters():
                if 'img' in name:
                    param.requires_grad = False
            for param in self.pts_fusion_layer.cam_lss.parameters():
                param.requires_grad = False

            self.img_backbone.apply(fix_bn)
            self.img_neck.apply(fix_bn)

        from mmengine.logging import MMLogger
        logger: MMLogger = MMLogger.get_current_instance()
        for name, param in self.named_parameters():
            if param.requires_grad is True:
                logger.info(name)

    def extract_img_feat(self, x, ) -> torch.Tensor:
        B, N, C, H, W = x.size()

        # with torch.autocast(device_type='cuda'):
        # x = x.reshape(B, -1, 6, C, H, W)

        # x_grad = x[:, :self.stop_prev_grad]
        # x_nograd = x[:, self.stop_prev_grad:]
        # x_grad = x_grad.reshape(-1, C, H, W)
        # x_nograd = x_nograd.reshape(-1, C, H, W)
        x = x.reshape(B * N, C, H, W)
        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        with torch.autocast('cuda'):
            x = self.img_backbone(x.half())
            x = self.img_neck(x)
            x = list(x)

        # with torch.no_grad():
        #     self.eval()
        #     x_nograd = self.img_backbone(x_nograd)
        #     self.train()
        # if isinstance(x, dict):
        #     x = list(x.values())

        img_feats_reshaped = []
        for img_feat in x:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W).float())

        # for lvl, feat in enumerate(img_feats_reshaped):
        #     B, TN, C, H, W = feat.shape  # [B, TN, GC, H, W]
        #     N, T, = 6, TN // 6,
        #     feat = feat.reshape(B, T, N, C, H, W)
        #     img_feats_reshaped[lvl] = feat.contiguous()

        return img_feats_reshaped

    def sampling_pillar_heights(self, coords):
        batch_size = coords[-1, 0].item() + 1
        out_size_factor = self.test_cfg["pts"]["out_size_factor"]
        grid_size = self.test_cfg["pts"]["grid_size"]
        pc_range = self.test_cfg["pts"]["pc_range"]
        voxel_size = self.test_cfg["pts"]["voxel_size"]
        grid_voxel = torch.zeros([batch_size, grid_size[0] // out_size_factor, grid_size[1] // out_size_factor, grid_size[2]], device=coords.device)
        batch_idx = coords[..., 0].long()
        x_ids = (coords[..., 1] / out_size_factor).long()
        y_ids = (coords[..., 2] / out_size_factor).long()
        z_idx = (coords[..., 3]).long()
        grid_voxel[batch_idx, x_ids, y_ids, z_idx] = coords[..., 3].float()
        sample_height_max = (grid_voxel.max(dim=-1, keepdim=True)).values
        grid_voxel[grid_voxel == 0] = 100
        sample_height_min = (grid_voxel.min(dim=-1, keepdim=True)).values
        sample_height_min[sample_height_min == 100] = 0

        # torch.nonzero(sample_height_min[..., 0])
        sample_height_c = (sample_height_max + sample_height_min) / 2
        sample_height_h = (sample_height_max - sample_height_min).float() + 1

        ref_point_pos = torch.nonzero(sample_height_c[..., 0])
        ref_xy = (ref_point_pos[:, 1:] + 0.5) * voxel_size[0] * out_size_factor + pc_range[0]
        ref_z = sample_height_c[ref_point_pos[:, 0], ref_point_pos[:, 1], ref_point_pos[:, 2]]
        ref_h = sample_height_h[ref_point_pos[:, 0], ref_point_pos[:, 1], ref_point_pos[:, 2]]
        ref_z = ref_z / grid_size[2] * 8
        ref_h = ref_h / grid_size[2] * 8
        ref_z = ref_z - 5
        ref_wl = torch.ones_like(ref_xy) * voxel_size[0] * out_size_factor
        ref_point = torch.cat([ref_xy, ref_z, ref_wl, ref_h], dim=-1)
        return ref_point, ref_point_pos
        # fig, ax = plt.subplots()
        # plt.axis("off")
        # cax=ax.imshow(sample_height_max.transpose((1,2,0)))

    def generateDepth(self, batch_inputs_dict, batch_input_metas):
        points = batch_inputs_dict['points']
        img_size = batch_input_metas[0]["pad_shape"]
        batch_size = len(points)
        depth = torch.zeros(batch_size, 6, 2, img_size[0], img_size[1]).to(points[0].device)  # 创建大小 [b, n, 1, 448, 800]
        # img_lidar = torch.zeros(batch_size, 6, 5, img_size[0], img_size[1]).to(points[0].device)
        lidar2img = batch_input_metas[0]['lidar2img']
        img_aug_matrix = batch_input_metas[0]['img_aug_matrix']
        lidar_aug_matrix = batch_input_metas[0]['lidar_aug_matrix']
        lidar_aug_matrix_expanded = lidar_aug_matrix[:, None].expand(*lidar2img.shape)
        lidar_aug_matrix_expanded_inv = torch.inverse(lidar_aug_matrix_expanded)
        final_lidar2img = torch.matmul(lidar2img, lidar_aug_matrix_expanded_inv)
        # final_lidar2img = img_aug_matrix.matmul(lidar2img)
        for b in range(batch_size):

            cur_coords = points[b][:, :3]  # 取点的xyz

            p_num = cur_coords.shape[0]
            cur_coords = cur_coords[:, None, :]
            ones = torch.ones_like(cur_coords[..., :1])

            cur_coords = torch.cat([cur_coords, ones], dim=-1)
            cur_coords = cur_coords[:, :, :, None]
            cur_coords = cur_coords.expand(p_num, 6, 4, 1)

            lidar2img = final_lidar2img[b:b+1].expand(p_num, 6, 4, 4)
            cur_coords = torch.matmul(lidar2img, cur_coords)

            eps = 1e-5
            homo = cur_coords[..., 2:3, :]
            homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
            homo_nonzero = torch.clamp(homo_nonzero, 1e-4, 1e4)
            cur_coords[..., 0:2, :] = cur_coords[..., 0:2, :] / homo_nonzero

            img_aug = img_aug_matrix[b:b + 1].expand(p_num, 6, 4, 4)
            cur_coords = torch.matmul(img_aug, cur_coords).squeeze(-1)
            homo = homo.squeeze(-1)
            valid_mask = ((homo > eps) \
                          & (cur_coords[..., 1:2] > 0.0)
                          & (cur_coords[..., 1:2] < img_size[0])
                          & (cur_coords[..., 0:1] > 0.0)
                          & (cur_coords[..., 0:1] < img_size[1])
                          ).squeeze(-1)

            for c in range(6):
                masked_coords = cur_coords[valid_mask[:, c], c].long()  # 点云投影到图像坐标
                masked_dist = homo[valid_mask[:, c], c, 0]  # 对应深度
                # cur_p = cur_points[valid_mask[:, c], c, :].transpose(0, 1)
                depth[b, c, 0, masked_coords[:, 1], masked_coords[:, 0]] = masked_dist
                depth[b, c, 1, masked_coords[:, 1], masked_coords[:, 0]] = 1
                # img_lidar[b, c, :, masked_coords[:, 1], masked_coords[:, 0]] = cur_p.float()

        batch_input_metas[0]["gt_depth"] = torch.round(depth[:, :, 0].clone())
        depth_mean = 14.41
        depth_std = 156.89
        depth[:, :, 0] = (depth[:, :, 0] - depth_mean) / math.sqrt(depth_std)
        depth[:, :, 0] = depth[:, :, 0] * depth[:, :, 1]
        batch_input_metas[0]["sparse_depth"] = depth
        # batch_input_metas[0]["img_lidar"] = img_lidar

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        # with torch.autocast('cuda', enabled=False):
        points = [point.float() for point in points]
        feats, coords, sizes = self.voxelize(points)
        ref_point, ref_point_pos = self.sampling_pillar_heights(coords)
        batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x, ref_point, ref_point_pos

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feat, pst_feat = self.extract_feat(batch_inputs_dict, batch_input_metas)
        with torch.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
            new_pts_feat, new_img_feat = self.pts_fusion_layer(img_feat, pst_feat, batch_input_metas)
        if self.pts_bbox_head:
            outs = self.pts_bbox_head(new_pts_feat, img_inputs=new_img_feat, img_metas=batch_input_metas, gt_bboxes_3d=None, gt_labels_3d=None)

            outputs = self.pts_bbox_head.get_bboxes(outs, batch_data_samples, rescale=False)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
            self,
            batch_inputs_dict,
            batch_input_metas,
            **kwargs,
    ):
        imgs = batch_inputs_dict.get('img', None)
        points = batch_inputs_dict.get('points', None)
        # plt.imshow(imgs[0, 0,].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()
        batch_input_metas[0]['imgs'] = imgs
        pts_feature = None
        img_feature = None

        if self.input_img:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

            batch_input_metas[0]['lidar2img'] = lidar2image
            batch_input_metas[0]['img_aug_matrix'] = img_aug_matrix
            batch_input_metas[0]['lidar_aug_matrix'] = lidar_aug_matrix
            img_feature = self.extract_img_feat(imgs)
        with torch.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
            self.generateDepth(batch_inputs_dict, batch_input_metas)

        # fig, ax = plt.subplots()
        # plt.axis("off")
        # mask = batch_input_metas[0]["gt_depth"][0, 0:1].permute(1, 2, 0).detach().cpu()
        # mask1 = torch.nonzero(mask)
        # image = imgs[0, 0]
        # image = image.permute(1, 2, 0).detach().cpu()
        # mean = [123.675, 116.28, 103.53]
        # std = [58.395, 57.12, 57.375]
        # image[..., 0] = image[..., 0] * std[0] + mean[0]
        # image[..., 1] = image[..., 1] * std[1] + mean[1]
        # image[..., 2] = image[..., 2] * std[2] + mean[2]
        # # image = torch.zeros_like(image)
        # image = torch.ones_like(image)*255
        # # image = image.int().numpy()
        # plt.imshow(image)
        # # plt.imshow(image)
        # ax.scatter(mask1[:, 1], mask1[:, 0], linewidths=0, c=mask[mask1[..., 0], mask1[..., 1]].numpy(), s=1)
        # # ax.scatter(mask1[:, 1], mask1[:, 0], linewidths=0, c='white', s=1)
        # plt.savefig("4.2.png", dpi=1200)
        # # plt.show()
        # plt.clf()
        with torch.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
            if self.input_pts:
                pts_feature, ref_point, ref_point_pos = self.extract_pts_feat(batch_inputs_dict)
                batch_input_metas[0]['ref_point'] = ref_point
                batch_input_metas[0]['ref_point_pos'] = ref_point_pos
                pts_feature = self.pts_backbone(pts_feature)
                pts_feature = self.pts_neck(pts_feature)
        return pts_feature, img_feature

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_gt_instances_3d = [ds.gt_instances_3d for ds in batch_data_samples]
        # batch_inputs_dict['points'][0]
        gt_bboxes_3d = [
            gt_instances.bboxes_3d for gt_instances in batch_gt_instances_3d
        ]
        gt_labels_3d = [
            gt_instances.labels_3d for gt_instances in batch_gt_instances_3d
        ]

        pts_feats, img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        with torch.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
            new_pts_feat, new_img_feat = self.pts_fusion_layer(pts_feats, img_feats, batch_input_metas)
            if self.pts_bbox_head:
                outs = self.pts_bbox_head(new_pts_feat, img_inputs=new_img_feat, img_metas=batch_input_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
                losses = self.pts_bbox_head.loss(*loss_inputs)
                if 'loss_depth' in batch_input_metas[0].keys():
                    losses['loss_depth'] = batch_input_metas[0]['loss_depth']

        losses.update(losses)

        return losses

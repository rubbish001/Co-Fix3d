_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/cyclic-20e.py'
]
custom_imports = dict(
    imports=['projects.Co-Fix3d.co-fix3d'], allow_failed_imports=False)

voxel_size = [0.075, 0.075, 0.2]
# grid_size = [1536, 1536, 41]
grid_size = [1440, 1440, 41]
# point_cloud_range = [-57.6, -57.6, -5.0, 57.6, 57.6, 3.0]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# pos_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
backend_args = None
metainfo = dict(classes=class_names)

ida_aug_conf = {
    'resize_lim': (0.57, 0.825),
    'final_dim': (384, 1056),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (-5.4, 5.4),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}
# ida_aug_conf = {
#     'resize_lim': (0.38, 0.55),
#     'final_dim': (448, 800),
#     'bot_pct_lim': (0.0, 0.0),
#     'rot_lim': (-5.4, 5.4),
#     'H': 900,
#     'W': 1600,
#     'rand_flip': True,
# }
dataset_type = 'NuScenesDataset'
# dataset_type = 'CustomNuScenesDataset'
data_root = '/data3/li/data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')
input_modality = dict(use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False)
multistage_heatmap = 1
inter_channel = 128
step=2
extra_feat = None
input_img = True
input_pts = False
model = dict(
    type='CoFix3D',
    freeze_pts=True,
    freeze_img=False,
    input_img=input_img,
    input_pts=input_pts,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.075, 0.075, 0.2],
            max_voxels=[120000, 160000],
            voxelize_reduce=True)),

    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=grid_size,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,  128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_fusion_layer=dict(
        type='CoFix3DEncoder',
        hidden_channel=inter_channel,
        in_channels_pts=sum([256, 256]),
        bn_momentum=0.1,
        step=step,
        pc_range=point_cloud_range,
        img_scale=(ida_aug_conf['final_dim'][0], ida_aug_conf['final_dim'][1]),
        input_img=True,
        input_pts=False,
    ),
    pts_bbox_head=dict(
        type='CoFix3DDecoder',
        iter_heatmap=1,
        # reuse_first_heatmap=True,
        extra_feat=extra_feat,
        roi_feats=0, #7
        # roi_dropout_rate=0.1,
        roi_based_reg=False, # True
        # roi_expand_ratio=1.2,
        multiscale=True,
        multistage_heatmap=multistage_heatmap,
        mask_heatmap_mode='poscls',
        input_img=input_img,
        input_pts=input_pts,
        step=step,
        iterbev_wo_img=True,
        add_gt_groups=0,
        add_gt_groups_noise='box,1',
        add_gt_groups_noise_box='gtnoise',
        add_gt_pos_thresh=5.,
        add_gt_pos_boxnoise_thresh=0.75,
        gt_center_limit=5,
        bevpos=True,
        loss_weight_heatmap=1.,
        loss_weight_separate_heatmap=0.,
        loss_weight_separate_bbox=0.3,
        num_proposals=200,
        hidden_channel=inter_channel,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=8,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.15),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        decoder_cfg=dict(
            type='DeformableDetrTransformerDecoder',
            num_layers=3,
            return_intermediate=False,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=inter_channel,
                        num_heads=8,
                        dropout=0.1),
                    dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=inter_channel,
                        num_levels=3,
                        num_points=4,
                        num_heads=8,)
                ],
                feedforward_channels=1024,
                ffn_dropout=0.1,
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=inter_channel,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')))
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            grid_size=[1440, 1440, 41],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)), )),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 41],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-54.0, -54.0],
            nms_type=None)),
)

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        ida_aug_conf=ida_aug_conf, is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'lidar_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ImageAug3D',
        ida_aug_conf=ida_aug_conf, is_train=False),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx', 'lidar_aug_matrix',
            'lidar_path', 'img_path', 'num_pts_feats', 'num_views'
        ])
]

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=False,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))
    # dataset=dict(
    #     type='CBGSDataset',
    #     dataset=dict(
    #         type=dataset_type,
    #         data_root=data_root,
    #         ann_file='nuscenes_infos_train.pkl',
    #         pipeline=train_pipeline,
    #         metainfo=metainfo,
    #         modality=input_modality,
    #         test_mode=False,
    #         data_prefix=data_prefix,
    #         use_valid_flag=True,
    #         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    #         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    #         box_type_3d='LiDAR')))
val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    # drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    jsonfile_prefix='./work_dirs/diffusionBEV_lidar/12.15.8/results_eval',
    metric='bbox',
    backend_args=backend_args)
# test_evaluator = val_evaluator
test_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    modality=input_modality,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    # format_only=True,
    jsonfile_prefix='./work_dirs/diffusionBEV_lidar/12.15.8.1/results_eval1',
    metric='bbox',
    backend_args=backend_args)
num_epochs = 20
# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=20)
val_cfg = dict()
test_cfg = dict()
lr = 0.00001
optim_wrapper = dict(
    # type='OptimWrapper',
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

log_processor = dict(window_size=50)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)
# custom_hooks = [
#                 # dict(type='DisableObjectSampleHook', disable_after_epoch=1)
#                 # dict(type='LoadMyCheckpointHook', load_img_from='/data3/li/workspace/mm3d/cpt/vov_det_final.pth')
#                 ]
# load_from = '/home/li/workspace/mmdet3d/cpt/DeformFormer3D_C_R50_ep20.pth'
# auto_scale_lr = dict(enable=True, base_batch_size=16)
load_from = '/data3/li/workspace/mm3d/cpt/epoch_20.pth'
# auto_scale_lr = dict(enable=True, base_batch_size=16)
# load_from = '/home/li/workspace/mmdet3d/projects/PETR/configs/fcos3d_vovnet_imgbackbone-remapped.pth'
find_unused_parameters=True
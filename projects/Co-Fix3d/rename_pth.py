import torch
from collections import OrderedDict
import os
import torch.nn as nn
import torch.nn.init as init


def init_weight(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal(0, 0.01)
            m.bias.data.zero_()


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.find("img_backbone") > -1:
            name = k.replace("img_backbone.", "")
        #     new_state_dict[name] = v
        # else:
            new_state_dict[name] = v
        else:
            continue
    return new_state_dict


# 加载pretrain model
state_dict = torch.load('/data3/li/workspace/mm3d/cpt/epoch_20.pth')
new_dict = state_dict["state_dict"]
new_state_dict = OrderedDict()
for k, v in new_dict.items():
    if k.find("pts_bbox_head.heatmap_head.") > -1:
        new_state_dict[k] = v
        name1 = k.replace("pts_bbox_head.heatmap_head.0", "pts_bbox_head.heatmap_head.1")
        new_state_dict[name1] = v.clone()
        name2 = k.replace("pts_bbox_head.heatmap_head.0", "pts_bbox_head.heatmap_head.2")
        new_state_dict[name2] = v.clone()

    if k.find("pts_bbox_head.inpaint_bev.0") > -1:
        new_state_dict[k] = v
        name1 = k.replace("pts_bbox_head.inpaint_bev.0", "pts_bbox_head.inpaint_bev.1")
        new_state_dict[name1] = v.clone()
        name2 = k.replace("pts_bbox_head.inpaint_bev.0", "pts_bbox_head.inpaint_bev.2")
        new_state_dict[name2] = v.clone()
    else:
        new_state_dict[k] = v

state_dict["state_dict"] =new_state_dict
# torch.save(state_dict["state_dict"], '/home/li/workspace/mmdet3d/cpt/bevfusion_lidar_voxel0075_second_secfpn_8xb4.pth')

# keys = []
# for k, v in new_dict.items():
#     if k.startswith('conv_cls'):  # 将‘conv_cls’开头的key过滤掉，这里是要去除的层的key
#         continue
#     keys.append(k)
#
# # 去除指定层后的模型
# new_dict = {k: new_dict[k] for k in keys}

# net = VoVNet(spec_name='V-99-eSE', norm_eval=True, frozen_stages=-1, input_ch=3, out_features=(
#             'stage2',
#             'stage3',
#             'stage4',
#             'stage5',
#         ),)  # 自己定义的模型，但要保证前面保存的层和自定义的模型中的层一致

# 加载pretrain model中的参数到新的模型中，此时自定义的层中是没有参数的，在使用的时候需要init_weight一下
# net.state_dict().update(state_dict["state_dict"])
# net.load_state_dict(new_dict, strict=True)
# 保存去除指定层后的模型
torch.save(state_dict, '/data3/li/workspace/mm3d/cpt/epoch_20_cp.pth')
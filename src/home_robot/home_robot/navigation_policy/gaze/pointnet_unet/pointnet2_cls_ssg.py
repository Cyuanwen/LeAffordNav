import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction

class PointNet2Wrapper(nn.Module):
    def __init__(self, num_output, ckpt_path, in_channel=4,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_output = num_output
        self.pointnet2 = get_model(in_channel=in_channel)
        # 由于增加了一个维度，无法使用预训练模型
        # ckpt = torch.load(ckpt_path)
        # self.pointnet2.load_state_dict(ckpt['model_state_dict'])

        # # 冻结参数
        # for name, param in self.pointnet2.named_parameters():
        #     # make the parameters not trainable
        #     if name.startswith("fc3") or name.startswith("fc2") or name.startswith("fc1") or name.startswith("sa3"):
        #         continue
        #     param.requires_grad = False

        # # # @cyw 最后一层的分类头不要
        # # pointnets2_modules = list(self.pointnet2.children())[:-1]
        # # self.pointnet2 = torch.nn.Sequential(*pointnets2_modules)
        # # 这样删除后，调用的forward函数变了

    def forward(self, points):
        '''
            input: B N C
            output: B 256
        '''
        with torch.no_grad():
            points = torch.permute(points, [0, 2, 1])
            feature, _ = self.pointnet2(points)
        return feature


class get_model(nn.Module):
    # def __init__(self, num_class=40, normal_channel=False):
    def __init__(self, in_channel=4, num_class=40, normal_channel=True):
        super(get_model, self).__init__()
        # in_channel = 6 if normal_channel else 3
        # in_channel = 6 if normal_channel else 4
        # in_channel = 4 
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.3, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, classify=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm) # bs 3 num_point and bs 128 num_point
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # 只需要特征，丢弃前向层
        # x = self.fc3(x)
        # if classify:
        #     x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

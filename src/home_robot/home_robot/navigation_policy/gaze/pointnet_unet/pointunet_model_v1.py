from turtle import forward
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


from .pointnet2_cls_ssg import PointNet2Wrapper
# from .core.registry import registry
from .encoders.fusion import FusionMult
from .encoders.resnet import ConvBlock, IdentityBlock
from .encoders.unet import Up,DoubleConv,Down

import clip
from clip.model import Bottleneck,AttentionPool2d


'''
PointUnet model: 
1. 使用pointnet提取 点云特征
2. 用clip image encoder提取 地图特征
3. 将两个特征融合起来，解码
TODO 看一下clip的网络结构，是否直接使用unet提取地图特征？

ref: seeing the unseen
https://github.com/Ram81/seeing-unseen

'''

"""image encoder"""
class ModifiedResNet(nn.Module):
    """
    ref: /home/tmn/anaconda3/envs/ovmm/lib/python3.9/site-packages/clip/model.py
    add input_channel

    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, input_dim=2):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(input_dim, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        # self.attnpool = AttentionPool2d(input_resolution // 5, embed_dim, heads, output_dim) # 160 / 5 = 32
        # NOTE self.attnpool 没有使用，可能会出现pytorch-lightning多线程训练报错

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

def forward_prepool(self, x):
    """
    Adapted from https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L138
    Expects a batch of images where the batch number is even. The whole batch
    is passed through all layers except the last layer; the first half of the
    batch will be passed through avgpool and the second half will be passed
    through attnpool. The outputs of both pools are concatenated returned.
    """

    im_feats = []
    def stem(x):
        for conv, bn, relu in [(self.conv1, self.bn1, self.relu1), (self.conv2, self.bn2, self.relu2), (self.conv3, self.bn3, self.relu3)]:
            x = relu(bn(conv(x)))
            im_feats.append(x)
        x = self.avgpool(x)
        # im_feats.append(x) # 只用到 -2，-3，-4
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)

    for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
        x = layer(x)
        im_feats.append(x)
    # 没有attention pool
    # @cyw
    # x = self.attnpool(x) # 32 2048 5 5 -> 32 2048
    return x, im_feats

class Unet_encoder(nn.Module):
    """Unet encoder
    """
    def __init__(self, n_channels):
        super(Unet_encoder, self).__init__()
        self.n_channels = n_channels

        self.inc = (DoubleConv(n_channels, 64)) # channel 变小，但图像尺寸和原来一样
        self.down1 = (Down(64, 128)) # channel， 图像尺寸变小一半
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.down5 = (Down(1024, 2048))
    
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        im_feats = []
        im_feats.append(x2)
        im_feats.append(x3)
        im_feats.append(x4)
        im_feats.append(x5)
        im_feats.append(x6)
        return x6, im_feats


class PointUNet(nn.Module):
    def __init__(
        self, 
        map_encoder_type:str="resnet",
        map_encoder_pretrain:bool=True,
        map_encoder_trainable:bool=False,
        input_resolution:int=160,
        map_output_dim: int=2048,
        map_layers:Tuple=(3,4,6,3),
        pcd_num_output:int=40,
        ckp_path:str="data/pretrained_checkpoint/pointnet2_ssg_wo_normals_checkpoints/best_model.pth",
        upsample_factor:int=2,
        bilinear: bool = True,
        batchnorm: bool = True,
        output_dim: int = 1,
        *args, 
        **kwargs
    ) -> None:
        """ PointNet Unet model
            Input:
                map_encoder_params:
                    map_encoder_type:str="resnet", 
                        choice from resnet and unet
                    map_encoder_pretrain:bool=True, 
                        是否使用预训练的 encoder,仅当 map_encoder_type 为 resnet时有效
                    map_encoder_trainable:bool=False,
                        是否在原本模型上训练
                    input_resolution:int=160,
                    map_output_dim: int=2048,
                    map_layers:Tuple=(3,4,6,3),
                        resnet bottleneck层数，仅当 map_encoder_type 为 resnet时有效
        """
        super().__init__(*args, **kwargs)

        # map encoder
        self.resize_transforms = None
        if map_encoder_type == "resnet":
            if map_encoder_pretrain:
                model, preprocess = clip.load("RN50")
                map_encoder = model.visual
                map_encoder.conv1 = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                map_encoder.conv1.weight = nn.Parameter(map_encoder.conv1.weight.data.half())
                # 原本模型 (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                # resize
                resize_transforms = []
                # expected input: H x W x C (np.uint8 in [0-255])
                if input_resolution != 224 :
                    print("Using CLIP preprocess for resizing+cropping to 224x224")
                    resize_transforms = [
                        # resize and center crop to 224
                        preprocess.transforms[0],
                        preprocess.transforms[1],
                    ]

                self.resize_transforms = T.Compose(resize_transforms)
                if map_encoder_trainable:
                    map_encoder.to(torch.float32)
                else:
                    for param in map_encoder.parameters():
                        param.requires_grad = False
                    map_encoder.eval()
            else:
                vision_width = 64
                vision_heads = vision_width * 32 // 64
                map_encoder = ModifiedResNet(
                    layers= map_layers,
                    output_dim=map_output_dim,
                    heads=vision_heads,
                    input_resolution=input_resolution,
                )
            # Overwrite forward method to return both attnpool and avgpool
            # concatenated together (attnpool + avgpool).
            bound_method = forward_prepool.__get__(
                map_encoder, map_encoder.__class__
            )
            setattr(map_encoder, "forward", bound_method)

        elif map_encoder_type == 'unet':
            if not map_encoder_pretrain:
                map_encoder = Unet_encoder(2) # bilinear 为false，输出为 1024 channel
            else:
                raise NotImplementedError

        self.map_encoder_type = map_encoder_type 
        self.map_encoder = map_encoder
        self.map_encoder_out_dim = 2048
        self.map_encoder_half = map_encoder_type == "resnet" and map_encoder_pretrain and not map_encoder_trainable

        # pcd encoder
        self.pcd_encoder = PointNet2Wrapper(
            pcd_num_output,
            ckp_path,
            in_channel=3
        )
        # def __init__(self, num_output, ckpt_path, *args, **kwargs):
        self.pcd_encoder_out_dim = 1024
        
        # decoder
        self.upsample_factor = upsample_factor
        self.bilinear = bilinear
        self.batchnorm = batchnorm
        self.output_dim = output_dim
        self.init_decoder()
    
    def init_decoder(self):
        # TODO
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.map_encoder_out_dim, 1024, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )

        self.lang_fuser1 = FusionMult(input_dim=self.map_encoder_out_dim // 2)
        self.lang_fuser2 = FusionMult(input_dim=self.map_encoder_out_dim // 4)
        self.lang_fuser3 = FusionMult(input_dim=self.map_encoder_out_dim // 8)

        self.lang_proj1 = nn.Linear(self.pcd_encoder_out_dim, 1024)
        self.lang_proj2 = nn.Linear(self.pcd_encoder_out_dim, 512)
        self.lang_proj3 = nn.Linear(self.pcd_encoder_out_dim, 256)

        self.up1 = Up(
            self.map_encoder_out_dim, 1024 // self.upsample_factor, self.bilinear
        )
        self.up2 = Up(1024, 512 // self.upsample_factor, self.bilinear)
        self.up3 = Up(512, 256 // self.upsample_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(
                128,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                64,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(
                64,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                32,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(
                32,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                16,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )
    
    def forward_encoder(self, map, pcd_feature):
        if self.resize_transforms is not None:
            map = self.resize_transforms(map)
            # batch = torch.stack(
            #     [self.resize_transforms(img) for img in batch]
            # )
        x, x_im_feats = self.map_encoder(map)

        x = x.to(torch.float32)
        # x_im_feats = x_im_feats.to(torch.float32)

        x = self.conv1(x)

        x = self.lang_fuser1(x, pcd_feature, x2_proj=self.lang_proj1)
        x = self.up1(x, x_im_feats[-2])

        x = self.lang_fuser2(x, pcd_feature, x2_proj=self.lang_proj2)
        x = self.up2(x, x_im_feats[-3])

        x = self.lang_fuser3(x, pcd_feature, x2_proj=self.lang_proj3)
        x = self.up3(x, x_im_feats[-4])
        return x

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        '''
        data format:
            {
            'map': map_data,
            'pcd_coords': data['pcd_base_coord_s'],
            'output': data['target_s']
            }
        map_data: b 3 h w
        pcd_coords: b n c
        '''
        pcd = batch["pcd_coords"]
        if len(pcd.shape)>3:
            pcd = pcd.squeeze()
        receptacle = batch["map"]

        # # debug
        # # 判断输入数据是否有nan
        # if torch.isnan(pcd).any() or torch.isnan(receptacle).any():
        #     print("debug")

        input_shape = receptacle.shape

        # @cyw
        pcd_feature = self.pcd_encoder(pcd)  # b * 256
        # # debug
        # if torch.isnan(pcd_feature).any():
        #     print("debug")
        
        x = self.forward_encoder(receptacle, pcd_feature)
        # # debug
        # if torch.isnan(x).any():
        #     print("debug")

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # # debug
        # if torch.isnan(x).any():
        #     print("debug")

        x = self.conv2(x)
        x = F.interpolate(
            x, size=(input_shape[-2], input_shape[-1]), mode="bilinear"
        )
        # TODO 这里是取logit后的值吗？
        # 看起来seeing unseen没有进行额外处理
        # # debug
        # if torch.isnan(x).any():
        #     print("debug")
        return x
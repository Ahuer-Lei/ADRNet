from .layers import *
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import kornia.utils as KU
import kornia.filters as KF
from copy import deepcopy
import os
import yaml
import numpy as np


class ShareFeature(nn.Module):
    def __init__(self):
        super(ShareFeature, self).__init__()
        feature_extractor = nn.ModuleList([])

        feature_extractor.append(Conv2d(1,3,kernel_size=3,stride=1,padding=1,dilation=1, norm=nn.InstanceNorm2d))
        feature_extractor.append(ResConv2d(3,3,kernel_size=3,stride=1,padding=2,dilation=2, norm=nn.InstanceNorm2d))
        feature_extractor.append(Conv2d(3,6,kernel_size=3,stride=1,padding=2,dilation=2, norm=nn.InstanceNorm2d))
        feature_extractor.append(ResConv2d(6,6,kernel_size=3,stride=1,padding=4,dilation=4, norm=nn.InstanceNorm2d))    
        feature_extractor.append(nn.Sequential(
            nn.Conv2d(6,1,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)         
            )
            )
        self.layers = feature_extractor   

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x


class FeatureEnhancement(nn.Module):
    def __init__(self, inchannel):
        super(FeatureEnhancement, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=4, padding=4),      # 16
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=8, padding=8),      # 32
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=16, padding=16),    # 64
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=32, padding=32),    #128
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=64, padding=64)     #256
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=4, padding=4),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=8, padding=8),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=16, padding=16),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=32, padding=32),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=2, padding=2),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=4, padding=4),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=8, padding=8),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=16, padding=16),
        )
        self.branch_4 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=1, padding=1),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=4, padding=4),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=8, padding=8),
        )

        # self.branch_5 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=1, padding=1),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=4, padding=4),

        # )

        # self.branch_6 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=1, padding=1),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2),
        # )
        # self.branch_7 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=1, padding=1)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        # x5 = self.branch_5(x)
        # x6 = self.branch_6(x)
        # x7 = self.branch_7(x)

        y = x + x1 + x2 + x3 + x4
        return y



class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).cuda()
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=True), flow



def normMask(mask, strength=0.5):
    """
    :return: to attention more region
    """
    batch_size, c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(batch_size, -1).max(1)[0]
    max_value = max_value.reshape(batch_size, 1, 1, 1)
    mask = mask/(max_value*strength)
    mask = torch.clamp(mask, 0, 1)

    return mask


# ResNet18/34的残差结构
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.down_sample is not None:
            residual = self.down_sample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)

        return out


class SPAtt(nn.Module):
    def __init__(self, in_ch):
        super(SPAtt, self).__init__()
        self.dilated_conv1_1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=1, padding=1)
        self.dilated_conv1_2 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=2, padding=2)
        self.dilated_conv1_3 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=4, padding=4)

        self.dilated_conv2_1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=1, padding=1)
        self.dilated_conv2_2 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=2, padding=2)
        self.dilated_conv2_3 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=4, padding=4)

        self.dilated_conv3_1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=1, padding=1)
        self.dilated_conv3_2 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=2, padding=2)
        self.dilated_conv3_3 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, dilation=4, padding=4)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        # line 1
        x1 = self.dilated_conv1_1(x)
        x2 = x + x1
        x3 = self.dilated_conv1_2(x2)
        x4 = x2 + x3
        x5 = self.dilated_conv1_3(x4)  # [1,512,16,16]
        x5 = self.relu(x5)

        # line 2
        y1 = self.dilated_conv2_1(x)
        y2 = x + y1
        y3 = self.dilated_conv2_2(y2)
        y4 = y2 + y3
        y5 = self.dilated_conv2_3(y4)  # [1,512,16,16]
        y5 = self.relu(y5)

        # line 3
        z1 = self.dilated_conv3_1(x)
        z2 = x + z1
        z3 = self.dilated_conv3_2(z2)
        z4 = z2 + z3
        z5 = self.dilated_conv3_3(z4)  # [1,512,16,16]
        z5 = self.relu(z5)

        x5 = x5.view(b, c, h*w).permute(0, 2, 1)   # [1,256,512]
        y5 = y5.view(b, c, h*w)   # [1,512,256]
        xy = torch.bmm(x5, y5)    # [1,256,256]
        attention = self.softmax(xy)  # [1,256,256]
        z5 = z5.view(b,c,h*w)     # [1,512,256]

        out = torch.bmm(z5, attention.permute(0,2,1))   # [1,512,256]
        out = out.view(b,c,h,w)   # [1,512,16,16]

        out = self.gamma * out + x

        return out

class ResNet(nn.Module):
    def __init__(self, block, block_num):
        super(ResNet, self).__init__()

        self.in_channel = 64

        self.conv1 = nn.Conv2d(2, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        self.feat_enhance = FeatureEnhancement(128) 
        self.conv_last = nn.Conv2d(512, 8, kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channel, block_num, stride=1):
        down_sample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, down_sample=down_sample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.feat_enhance(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.para_reg = resnet34()

        self.ShareFeature = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))

        self.genMask = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(512, 6)

    def forward(self, sar_stack, opt_stack):
        b,c,h,w = opt_stack.shape

        mask_sar = self.genMask(sar_stack)
        mask_opt = self.genMask(opt_stack)

        mask_sar = normMask(mask_sar)
        mask_opt = normMask(mask_opt)

        sar_feat = self.ShareFeature(sar_stack)
        opt_feat = self.ShareFeature(opt_stack)

        sar_feat_mask = torch.mul(sar_feat, mask_sar)   
        opt_feat_mask = torch.mul(opt_feat, mask_opt)

        x = torch.cat((sar_feat_mask, opt_feat_mask), dim=1)
        x = self.para_reg(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        y = torch.cat((opt_feat_mask, sar_feat_mask), dim=1)
        y = self.para_reg(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
  
        return x[0:int(b/2), ...], x[int(b/2):, ...], y[0:int(b/2), ...], y[int(b/2):, ...]



class DoubleConv(nn.Sequential):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        if mid_channel is None:
            mid_channel = out_channel
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channel,out_channel)
        )


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x//2, diff_x - diff_x // 2, diff_y//2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x



class UFeatureEnhancement(nn.Module):
    def __init__(self, inchannel):
        super(UFeatureEnhancement, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=4, padding=4),      # 16
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=2, padding=2),

        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=1, padding=1),

        )
        # self.branch_4 = nn.Sequential(
        #     nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, dilation=1, padding=1),
        # #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2),
        # #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=4, padding=4),
        # #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=8, padding=8),
        # )

        # self.branch_5 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=1, padding=1),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=4, padding=4),

        # )

        # self.branch_6 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=1, padding=1),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2),
        # )
        # self.branch_7 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=1, padding=1)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)

        # x5 = self.branch_5(x)
        # x6 = self.branch_6(x)
        # x7 = self.branch_7(x)

        y = x + x1 + x2 + x3
        return y


class OutConv(nn.Sequential):
    def __init__(self, in_channel, num_class):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channel, num_class, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self, in_channel: int = 2, num_class: int = 2, bilinear: bool = True, base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.num_class = num_class
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channel, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)

        factor = 2 if bilinear else 1

        self.down4 = Down(base_c*8, base_c*16//factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.spatt = SPAtt(512)
        self.u_featen = UFeatureEnhancement(512)
        self.process = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        )

    def forward(self, x):

        x1 = self.in_conv(x)  # [4,64,256,256]
        x2 = self.down1(x1)   # [4,128,128,128]
        x3 = self.down2(x2)   # [4,256,64,64]
        x4 = self.down3(x3)   # [4,512,32,32]
        x5 = self.down4(x4)   # [4,512,16,16]
        x5_1 = self.spatt(x5)
        x5_2 = self.u_featen(x5)
        x5 = x5_1 + x5_2
        x = self.up1(x5, x4)  # [4,256,32,32]
        x = self.up2(x, x3)   # [4,128,64,64]
        x = self.up3(x, x2)   # [4,64,128,128]
        x = self.up4(x, x1)   # [4,64,256,256]

        return x



class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.un = UNet()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.genMask = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )        
        self.process = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        )

    def forward(self, sar, opt):
        b,c,h,w = sar.shape

        sar_mask = normMask(self.genMask(sar))
        opt_mask = normMask(self.genMask(opt))
        sar_feat = self.feat(sar)
        opt_feat = self.feat(opt)
        sar_feat_mask = torch.mul(sar_feat, sar_mask)
        opt_feat_mask = torch.mul(opt_feat, opt_mask)

        u = torch.cat([sar_feat_mask, opt_feat_mask], dim=1)
        u = self.un(u)
        u = self.process(u)

        v = torch.cat([opt_feat_mask, sar_feat_mask], dim=1)
        v = self.un(v)
        v = self.process(v)

        disp = {'sar2rgb':u[0:int(b/2), ...], 'rgb2sar':v[int(b/2):, ...]}
        disp1 = {'sar2rgb':u[int(b/2):, ...], 'rgb2sar':v[0:int(b/2), ...]}

        return u,v,disp, disp1




























def get_scheduler(optimizer, opts, now_ep=-1):   # last
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / \
                float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=now_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=now_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler


def gaussian_weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and class_name.find('Conv') == 0:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass

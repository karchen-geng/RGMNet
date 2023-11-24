"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import mod_resnet
from model import cbam


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


# Single object version, used only in static image pretraining
# This will be loaded and modified into the multiple objects version later (in stage 1/2/3)
# See model.py (load_network) for the modification procedure
class ValueEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask, other_masks):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask, other_masks], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256

        x = self.fuser(x, key_f16)

        return x  # [4, 512, 24, 24]


class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)  # 1/4, 256
        f8 = self.layer2(f4)  # 1/8, 512
        f16 = self.layer3(f8)  # 1/16, 1024  [12, 1024, 24, 24]

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)


# 帧间attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                                     padding=padding, dilation=dilation, bias=False)

        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride=16, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, 1024, 1, bias=False)
        self.bn1 = BatchNorm(1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SRM(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRM, self).__init__()

        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Style pooling
        # AvgPool（全局平均池化）：
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        # StdPool（全局标准池化）
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)
        # Style integration
        # CFC（全连接层）
        z = self.cfc(u)  # (b, c, 1)
        # BN（归一化）
        z = self.bn(z)
        # Sigmoid
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)
        return x * g.expand_as(x)


# Bi-read Module (BRM) Bidirectional Read Module
class BidirectionalRead(nn.Module):
    def __init__(self, channel):
        super(BidirectionalRead, self).__init__()
        # project c-dimensional features to multiple lower dimensional spaces
        channel_low = channel // 16

        self.p_f1 = nn.Conv2d(channel, channel_low, kernel_size=1)  # p_f1: project image features
        self.p_f2 = nn.Conv2d(channel, channel_low, kernel_size=1)  # p_f2: project flow features

        self.c_f1 = nn.Conv2d(channel, 1, kernel_size=1)  # c_f1: transform image features to a map by conv 1x1
        self.c_f2 = nn.Conv2d(channel, 1, kernel_size=1)  # c_f2: transform flow features to a map by conv 1x1

        self.relu = nn.ReLU()

    # f1: t-1 image, f2: t image
    def forward(self, f1, f2):
        # Stack 1
        f1_1, f2_1 = self.forward_sa(f1, f2)  # soft attention
        f1_hat, f2_hat = self.forward_ca(f1_1, f2_1)  # co-attention
        fp1_hat = F.relu(f1_hat + f1)
        fp2_hat = F.relu(f2_hat + f2)

        # Stack 2
        f1_2, f2_2 = self.forward_sa(fp1_hat, fp2_hat)
        f1_hat, f2_hat = self.forward_ca(f1_2, f2_2)
        # fp1_hat = F.relu(f1_hat + fp1_hat)
        # fp2_hat = F.relu(f2_hat + fp2_hat)
        #
        # # Stack 3
        # f1_3, f2_3 = self.forward_sa(fp1_hat, fp2_hat)
        # f1_hat, f2_hat = self.forward_ca(f1_3, f2_3)
        # fp1_hat = F.relu(f1_hat + fp1_hat)
        # fp2_hat = F.relu(f2_hat + fp2_hat)
        #
        # # Stack 4
        # f1_4, f2_4 = self.forward_sa(fp1_hat, fp2_hat)
        # f1_hat, f2_hat = self.forward_ca(f1_4, f2_4)
        # fp1_hat = F.relu(f1_hat + fp1_hat)
        # fp2_hat = F.relu(f2_hat + fp2_hat)
        #
        # # Stack 5
        # f1_5, f2_5 = self.forward_sa(fp1_hat, fp2_hat)
        # f1_hat, f2_hat = self.forward_ca(f1_5, f2_5)
        #  [8, 512, 24, 24]
        return f1_2, f2_2

    # Soft Attention
    def forward_sa(self, f1, f2):
        c1 = self.c_f1(f1)  # channel -> 1
        c2 = self.c_f2(f2)  # channel -> 1

        n, c, h, w = c1.shape
        c1 = c1.view(-1, h * w)
        c2 = c2.view(-1, h * w)

        c1 = F.softmax(c1, dim=1)
        c2 = F.softmax(c2, dim=1)

        c1 = c1.view(n, c, h, w)
        c2 = c2.view(n, c, h, w)

        # Hadamard product
        f1_sa = c1 * f1
        f2_sa = c2 * f2

        # f1_sa and f2_sa indicate attention-enhanced features of t and t-1
        return f1_sa, f2_sa

    # f1: t-1 image, f2: t image
    def forward_ca(self, f1, f2):
        f1_cl = self.p_f1(f1)  # f1_cl: dimension from channel to channel_low
        f2_cl = self.p_f2(f2)  # f2_cl: dimension from channel to channel_low

        N, C, H, W = f1_cl.shape
        f1_cl = f1_cl.view(N, C, H * W)
        f2_cl = f2_cl.view(N, C, H * W)
        f2_cl = torch.transpose(f2_cl, 1, 2)

        # Affinity matrix
        A = torch.bmm(f2_cl, f1_cl)

        # softmax row and softmax col
        A_c = torch.tanh(A)
        A_r = torch.transpose(A_c, 1, 2)

        N, C, H, W = f1.shape

        f1_v = f1.view(N, C, H * W)
        f2_v = f2.view(N, C, H * W)

        # Attention
        f1_hat = torch.bmm(f1_v, A_r)
        f2_hat = torch.bmm(f2_v, A_c)
        f1_hat = f1_hat.view(N, C, H, W)
        f2_hat = f2_hat.view(N, C, H, W)

        f1_hat = F.normalize(f1_hat)
        f2_hat = F.normalize(f2_hat)

        return f1_hat, f2_hat

    def biread(self, affinity, mv, qv, mode=None):
        if mode == None:
            B, CV, T, H, W = mv.shape
            mo = mv.view(B, CV, T * H * W)
        elif mode == 'masked':
            B, CV, H, W = mv.shape
            mo = mv.view(B, CV, H * W)  # v_m 4*512*576
        mem = torch.bmm(mo, affinity)  # Weighted-sum B, CV, HW  4*512*576
        mem = mem.view(B, CV, H, W)  # [8, 512, 24, 24]
        # 不应该是简单的拼接
        en_mem, en_qv = self.forward(mem, qv)

        mem_out = torch.cat([en_mem, en_qv], dim=1)
        return mem_out

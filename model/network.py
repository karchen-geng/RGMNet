"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.compress = ResBlock(2048, 1024)
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()

    def get_affinity(self, mk, qk, mode=None):  # 4*64*1*24*24   4*64*24*24\
        if mode == None:
            B, CK, T, H, W = mk.shape
        elif mode == 'spatial':
            B, CK, H, W = mk.shape
        mk = mk.flatten(start_dim=2)  # 4*64*576
        qk = qk.flatten(start_dim=2)  # 4*64*576
        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk
        affinity = (2 * ab - a_sq) / math.sqrt(CK)  # B, THW, HW
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    def get_masked_affinity(self, affinity, mask):  # 4*64*24*24
        B, CV, H, W = mask.shape
        mo = mask.view(B, CV * H, W)  # v_m 4*576*576
        affinity = mo * affinity  # Weighted-sum B, C_H, W 4*576*576
        return affinity

    def readout(self, affinity, mv, qv, mode=None):
        if mode == None:
            B, CV, T, H, W = mv.shape
            mo = mv.view(B, CV, T * H * W)
        elif mode == 'masked':
            B, CV, H, W = mv.shape
            mo = mv.view(B, CV, H * W)  # v_m 4*512*576
        mem = torch.bmm(mo, affinity)  # Weighted-sum B, CV, HW  4*512*576
        mem = mem.view(B, CV, H, W)
        mem_out = torch.cat([mem, qv], dim=1)
        return mem_out


class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO()
        else:
            self.value_encoder = ValueEncoder()

            # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.aspp = ASPP(1024)
        # self.srm = SRM(1024)
        self.conv = nn.Conv2d(1025, 1, 3, 1, 1)
        self.memory = MemoryReader()
        self.biread = BidirectionalRead(512)
        self.decoder = Decoder()

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1 - prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))
        return logits

    def encode_key(self, frame):  # [4, 3, 3, 384, 384]
        # input: b*t*c*h*w
        b, t = frame.shape[:2]
        # 将一个batch内的所有帧提取特征 [12, 1024, 24, 24] [12, 512, 48, 48] [12, 256, 96, 96]
        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))  # [12, 3, 384, 384]
        k16 = self.key_proj(f16)  # key  [12, 64, 24, 24]
        f16_thin = self.key_comp(f16)  # [12, 512, 24, 24]

        # B*C*T*H*W   [4, 3, 64, 24, 24]  ->  [4, 64, 3, 24, 24]  -> [4, 64, 3, 24, 24]
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W  [4, 3, 512, 24, 24]
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])  # [4, 3, 512, 48, 48]
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None):
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2)  # B*512*T*H*W  [4, 512, 1, 24, 24]

    def segment(self, qk_16, qk16, qv16, qf8, qf4, mk16, mv16, Ms, sec_Ms = None, selector=None):
        # q - query,[4, 64, 24, 24],,[4, 512, 24, 24]   m - memory，[4, 64, 1, 24, 24],[4, 2, 512, 1, 24, 24],  Ms  4*1*384*384
        # qv16 is f16_thin above, qk_16 is t-1 frames
        # 和memory的相似度阵
        affinity = self.memory.get_affinity(mk16, qk16)  # 4*576*576
        # 和前一帧的相似度
        spatial_affinity = self.memory.get_affinity(qk_16, qk16, mode='spatial')  # 4*576*576

        # 相邻两帧的mem_out 与mask进行增强  mv16[:, 0][:, :, 0] 第一帧的第一个目标
        # if self.single_object: # ours
        #     mem_fir = self.memory.readout(spatial_affinity, mv16[:, :, 0], qv16, mode='masked')
        # else: # ours
        #     mem_fir = self.memory.readout(spatial_affinity, mv16[:, 0][:, :, 0], qv16, mode='masked')
        if self.single_object:
            # [8, 576, 576], [8, 512, 24, 24],[8, 512, 24, 24]
            mem_fir = self.biread.biread(spatial_affinity, mv16[:, :, 0], qv16, mode='masked')
        else:
            mem_fir = self.biread.biread(spatial_affinity, mv16[:, 0][:, :, 0], qv16, mode='masked')

        # mask-guide
        b, _, h, w = mem_fir.shape  # b*c*h*w
        mask_reshape_fir = F.interpolate(Ms, size=[h, w], mode='bilinear')
        concat_mem_fir = torch.cat([mem_fir, mask_reshape_fir], dim=1)  # B,C+1,H,W
        concat_mem_fir = torch.sigmoid(self.conv(concat_mem_fir))
        concat_mem_fir = mem_fir * concat_mem_fir
        # concat_mem_fir = self.srm(concat_mem_fir)
        mem_fir = self.aspp(concat_mem_fir)

        if self.single_object:
            # logits = self.decoder(torch.cat([mem_fir, self.memory.readout(affinity, mv16, qv16)], 1), qf8, qf4)
            logits = self.decoder(mem_fir + self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            # 第二个目标的mask-guide
            # mem_sec = self.memory.readout(spatial_affinity, mv16[:, 1][:, :, -1], qv16, mode='masked')  # ours
            mem_sec = self.biread.biread(spatial_affinity, mv16[:, 1][:, :, -1], qv16, mode='masked')

            mask_reshape_sec = F.interpolate(sec_Ms, size=[h, w], mode='bilinear')
            concat_mem_sec = torch.cat([mem_sec, mask_reshape_sec], dim=1)  # B,C+1,H,W
            concat_mem_sec = torch.sigmoid(self.conv(concat_mem_sec))
            concat_mem_sec = mem_sec * concat_mem_sec
            # concat_mem_sec = self.srm(concat_mem_sec)
            mem_sec = self.aspp(concat_mem_sec)

            mem_out_fir = mem_fir + self.memory.readout(affinity, mv16[:, 0], qv16)
            mem_out_sec = mem_sec + self.memory.readout(affinity, mv16[:, 1], qv16)  # 4*1024*24*24
            logits = torch.cat([
                self.decoder(mem_out_fir, qf8, qf4),
                # [4, 1, 384, 384]  4*1024*24*24
                self.decoder(mem_out_sec, qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]  # 归一后去除背景层

        return logits, prob

    def forward(self, mode, *args, **kwargs):  # 无名参数和字典
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

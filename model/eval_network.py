"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *
from model.network import Decoder


class STCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        # self.aspp = ASPP(1024)
        # self.srm = SRM(1024)
        # self.conv = nn.Conv2d(1025, 1, 3, 1, 1)
        self.biread = BidirectionalRead(512)
        # self.conv = nn.Conv2d(2, 1, 1, 1, 0)
        self.decoder = Decoder()

    def encode_value(self, frame, kf16, masks):
        k, _, h, w = masks.shape  # 2，1，h,w

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i != j]]
                    , dim=0, keepdim=True)
                for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame, kf16.repeat(k, 1, 1, 1), masks, others)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16, mask, key_k, key_v):
        k = mem_bank.num_objects

        readout_mem = mem_bank.match_memory(qk16)  # 两个key求相似度再与mem的v相乘
        readout_mem_last = mem_bank.match_last(qk16, key_k, key_v)  # 两个key求相似度再与mem的v相乘
        # 与mask增强
        # youtube required，davis is no required
        # if k == 1:
        #     mask = mask.unsqueeze(0)

        o, c, h, w = readout_mem_last.shape  # b*c*h*w
        mask_reshape = F.interpolate(mask, size=[h, w], mode='bilinear')

        qv16 = qv16.expand(k, -1, -1, -1)
        read = torch.cat([readout_mem, qv16], 1)

        #  bi-read
        # read_last = torch.cat([readout_mem_last, qv16], 1)
        en_readout_mem_last, en_qv16 = self.biread.forward(readout_mem_last, qv16)
        read_last = torch.cat([en_readout_mem_last, en_qv16], 1)


        concat_mem_fir = torch.cat([read_last, mask_reshape], dim=1)  # o,C+1,H,W
        concat_mem_fir = torch.sigmoid(self.conv(concat_mem_fir))
        concat_mem_fir = read_last * concat_mem_fir
        # read_last = self.aspp(self.srm(concat_mem_fir))
        read_last = self.aspp(concat_mem_fir)

        return torch.sigmoid(self.decoder(read + read_last, qf8, qf4))

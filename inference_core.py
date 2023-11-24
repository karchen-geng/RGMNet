import torch

from inference_memory_bank import MemoryBank
from model.eval_network import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by


class InferenceCore:
    def __init__(self, prop_net: STCN, images, num_objects, top_k=20, mem_every=5, include_last=False):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k + 1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh // 16
        self.kw = self.nw // 16

        self.mem_bank = MemoryBank(k=self.k, top_k=top_k)

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:, idx].cuda())
        return result

    def do_pass(self, key_k, key_v, idx, end_idx, mask):
        self.mem_bank.add_memory(key_k, key_v)
        closest_ti = end_idx  # 69

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx + 1, closest_ti)  # 1~68
        end = closest_ti - 1
        pre_mask = mask
        for ti in this_range:
            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)
            out_mask = self.prop_net.segment_with_query(self.mem_bank, qf8, qf4, k16, qv16, pre_mask, key_k, key_v)  # ti帧的mask
            key_k = k16.unsqueeze(2)
            # key_v = qv16  # 不能这样
            key_v = self.prop_net.encode_value(self.images[:, ti].cuda(), qf16, out_mask)
            pre_mask = out_mask
            out_mask = aggregate(out_mask, keep_bg=True)  # 3,1,h,w
            self.prob[:, ti] = out_mask

            if ti != end:  # 是否添加mem中
                is_mem_frame = ((ti % self.mem_every) == 0)
                if self.include_last or is_mem_frame:  # 没有将前一帧作为temp
                    prev_value = self.prop_net.encode_value(self.images[:, ti].cuda(), qf16, out_mask[1:])  # 去除背景项
                    prev_key = k16.unsqueeze(2)
                    self.mem_bank.add_memory(prev_key, prev_value, is_temp=not is_mem_frame)  # 优先作为mem

        return closest_ti

    # mask 2,1,h,w  frame_idx  0  end_idx 69
    def interact(self, mask, frame_idx, end_idx):  # mask数组,0,len(数组)
        mask, _ = pad_divide_by(mask.cuda(), 16)  # 将，2，1，480，910 变成2，1， 480，912
        # prob 3,69,1,h,w
        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        # KV pair for the interacting frame 得到第一帧的k和v 还有f
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:, frame_idx].cuda(), qf16,
                                           self.prob[1:, frame_idx].cuda())  # 第一维度的第0个数据[1：] 也就是除了背景的mask
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx, mask)

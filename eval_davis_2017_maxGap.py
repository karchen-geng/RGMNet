import os
from os import path
import time
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore

from progressbar import progressbar


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/Jul28_14_36_s01_mask_0720_bench_400000.pth')
# parser.add_argument('--davis_path', default='/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Dataset/DAVIS-2017-trainval-480p/davis_max_gap')
parser.add_argument('--davis_path', default='/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Dataset/DAVIS-2017-test-dev-480p/davis_max_gap')  # test
parser.add_argument('--output', default='output/s01_mask_0720/DAVIS2017_test_maxGap_mem1')
# parser.add_argument('--split', help='val/testdev', default='val')
parser.add_argument('--split', help='val/testdev', default='testdev')  # 用于test
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=1, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
# palette = Image.open(path.expanduser(davis_path + '/Annotations/480p/bike-packing/00000.png')).getpalette()
palette = Image.open(path.expanduser(davis_path + '/Annotations/480p/aerobatics/00000.png')).getpalette() # 用于test-dev
torch.autograd.set_grad_enabled(False)

# Setup Dataset
if args.split == 'val':
    test_dataset = DAVISTestDataset(davis_path, imset='2017/val.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
elif args.split == 'testdev':
    test_dataset = DAVISTestDataset(davis_path, imset='2017/test-dev.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
else:
    raise NotImplementedError

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)
total_process_time = 0
total_frames = 0
# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb'].cuda()  # 1,69,3,h,w
        msk = data['gt'][0].cuda()  # 2,69,1,h,w
        info = data['info']
        name = info['name'][0]  # bike-packing
        k = len(info['labels'][0])  # 2
        size = info['size_480p']  # h,w
        torch.cuda.synchronize()
        process_begin = time.time()
        processor = InferenceCore(prop_model, rgb, k, top_k=top_k, 
                        mem_every=args.mem_every, include_last=args.include_last)
        processor.interact(msk[:,0], 0, rgb.shape[1])  # [:,0]取第一维度的零个数据！
        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        for ti in range(processor.t):  # 69
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)
        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)
        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]
        # Save the results
        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))
        del rgb
        del msk
        del processor
print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)




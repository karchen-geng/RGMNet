# RGMNet
## Requirements

We used these packages/versions in the development of this project. 
- PyTorch `1.8.1`
- torchvision `0.9.1`


### Inference
- `eval_davis.py` for DAVIS 2017 validation and test-dev set (controlled by `--split`)
- `eval_youtube.py` for YouTubeVOS 2018/19 validation set (controlled by `--yv_path`)

### Training

Pre-training on static images: `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=2 train.py --id s0 --stage 0`

Main training: `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=2 train.py --id s01 --load_network s0.pth  --stage 1



# RGMNet
## Requirements

We used these packages/versions in the development of this project. 
- PyTorch `1.8.1`
- torchvision `0.9.1`
`pip install progressbar2 opencv-python gitpython gdown git+https://github.com/cheind/py-thin-plate-spline`

## Results


### Inference

- `eval_davis.py` for DAVIS 2017 validation and test-dev set (controlled by `--split`)
- `eval_youtube.py` for YouTubeVOS 2018/19 validation set (controlled by `--yv_path`)


### Training


Pre-training on static images: `CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train.py --id retrain_s0 --stage 0`

Main training: `CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train.py --id retrain_s03 --load_network [path_to_trained_s0.pth]  --stage 3`

</details>


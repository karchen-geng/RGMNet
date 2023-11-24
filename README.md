# RGMNet
## Requirements

- PyTorch `1.8.1`
- torchvision `0.9.1`
- progressbar2
- [thinspline](https://github.com/cheind/py-thin-plate-spline) for training (`pip install git+https://github.com/cheind/py-thin-plate-spline`)


## Results

| Dataset | Split |  J&F | J | F |
| --- | --- | :--:|:--:|:---:|
| DAVIS 2017 | validation | 85.6 | 82.0 | 89.2 |
| DAVIS 2017 | test-dev | 78.1 | 74.3 | 81.8 |

| Dataset | Split | Overall Score |
| --- | --- | :--:|
| YouTubeVOS 18 | validation | 84.2 |
| YouTubeVOS 19 | validation | 83.7 |


## Outputs
- davis17: https://pan.baidu.com/s/1BBVRdhNAO5-kz4IsRGKp2g?pwd=8d5m 
- davis17-test：https://pan.baidu.com/s/1GQYdBe7K5ew_uCzji1z52g?pwd=udf6
  
- youtube18：https://pan.baidu.com/s/1xoWsjaaR7VD1Cyu34dm-YA?pwd=u72v 
- youtube19：https://pan.baidu.com/s/1F0C5kEFiKjWL9rOPDIwZcA?pwd=lzez 


### Pretrained models
- s01-DAVIS：https://pan.baidu.com/s/1Cflan6X3K4Q9btAoFgtSxA?pwd=a6j4 
- s01-YouTube：https://pan.baidu.com/s/1sq5YKJ3BCs1l4JbxBjKy2Q?pwd=gm4c 


## Inference
- `eval_davis.py` for DAVIS 2017 validation and test-dev set (controlled by `--split`)
- `eval_youtube.py` for YouTubeVOS 2018/19 validation set (controlled by `--yv_path`)

## Training

Pre-training on static images: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=2 train.py --id s0 --stage 0

Main training: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=2 train.py --id s01 --load_network s0.pth  --stage 1



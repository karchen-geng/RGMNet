import datetime
import os
from os import path
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed
from model.model import STCNModel
from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset
from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from util.load_subset import load_sub_davis, load_sub_yv
from util.spent_time import TimeSpent


"""
Initial setup
"""
# --Debug
# Init distributed environment
# os.environ['WORLD_SIZE'] = '1'
# os.environ['RANK'] = '0'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12321'


distributed.init_process_group(backend="nccl")
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)
print('CUDA Device count: ', torch.cuda.device_count())

# Parse command line arguments
para = HyperParameters()
para.parse()
if para['benchmark']:
    print('benchmark')
    torch.backends.cudnn.benchmark = True
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)
print('I am rank %d in this world of size %d!' % (local_rank, world_size))

"""
Model related
"""
if local_rank == 0:
    # Logging
    if para['id'].lower() != 'null':
        print('I will take the role of logging!')
        long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H:%M'), para['id'])
    else:
        long_id = None
    logger = TensorboardLogger(para['id'], long_id)
    logger.log_string('hyperpara', str(para))
    # Construct the rank 0 model
    model = STCNModel(para, logger=logger, 
                    save_path=path.join('saves', long_id, long_id) if long_id is not None else None, 
                    local_rank=local_rank, world_size=world_size).train()
else:
    # Construct model for other ranks
    model = STCNModel(para, local_rank=local_rank, world_size=world_size).train()

# Load pretrained model
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
    print('Previously trained model loaded!')
else:
    total_iter = 0
if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

"""
Dataloader related
"""
# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id): 
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)

def construct_loader(dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    train_loader = DataLoader(dataset, para['batch_size'], sampler=train_sampler, num_workers=para['num_workers'],
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    return train_sampler, train_loader

def renew_vos_loader(max_skip):
    yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'), 
                        path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=load_sub_yv())
    davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'), 
                        path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=load_sub_davis())
    train_dataset = ConcatDataset([davis_dataset]*5 + [yv_dataset])

    print('YouTube dataset size: ', len(yv_dataset))
    print('DAVIS dataset size: ', len(davis_dataset))
    print('Concat dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)
    return construct_loader(train_dataset)


"""
加载数据集
"""

skip_values = [10, 15, 20, 25, 5]
if para['stage'] == 0:
    static_root = path.expanduser(para['static_root'])
    fss_dataset = StaticTransformDataset(path.join(static_root, 'fss'), method=0)
    duts_tr_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TR'), method=1)
    duts_te_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TE'), method=1)
    ecssd_dataset = StaticTransformDataset(path.join(static_root, 'ecssd'), method=1)
    big_dataset = StaticTransformDataset(path.join(static_root, 'BIG_small'), method=1)
    hrsod_dataset = StaticTransformDataset(path.join(static_root, 'HRSOD_small'), method=1)
    # BIG and HRSOD have higher quality, use them more
    train_dataset = ConcatDataset([fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset]
             + [big_dataset, hrsod_dataset]*5)
    train_sampler, train_loader = construct_loader(train_dataset)
    print('Static dataset size: ', len(train_dataset))

elif para['stage'] == 1:
    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
    yv_root = path.join(path.expanduser(para['yv_root']), 'train')
    davis_root = para['davis_root']
    train_sampler, train_loader = renew_vos_loader(5)
    renew_loader = renew_vos_loader

"""
Determine current/max epoch
"""

total_epoch = math.ceil(para['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
print('len(train_loader): ', len(train_loader))
print('Number of training epochs (the last epoch might not complete): ', total_epoch)
time0 = TimeSpent()
time0.start_time()
if para['stage'] != 0:
    increase_skip_epoch = [round(total_epoch*f) for f in increase_skip_fraction]
    print('The skip value will increase approximately at the following epochs: ', increase_skip_epoch[:-1])

"""
Starts training
"""
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1) + local_rank*100)
try:  # 一个epoch 把每个视频都采样三帧，
    for e in range(current_epoch, total_epoch): 
        print('Epoch %d/%d' % (e, total_epoch))
        epoch_start_time = datetime.datetime.now()
        if para['stage'] != 0 and e != total_epoch and e>= increase_skip_epoch[0]:  # 制定epoch后跳帧？
            while e >= increase_skip_epoch[0]:
                cur_skip = skip_values[0]
                skip_values = skip_values[1:]
                increase_skip_epoch = increase_skip_epoch[1:]
            print('Increasing skip to: ', cur_skip)
            train_sampler, train_loader = renew_loader(cur_skip)  # 重新加载数据集  跳帧

        # Crucial for randomness! 
        train_sampler.set_epoch(e)
        # Train loop
        model.train()
        for data in train_loader:
            model.do_pass(data, total_iter)  # 训练一个batch
            total_iter += 1
            if total_iter >= para['iterations']:
                break
        epotch_time = datetime.datetime.now() - epoch_start_time
        str = "Epotch {}:总用时 {}天 {:.0f}小时 {:.0f}分钟 {:.4f}秒".format(e, epotch_time.days,
                                                                            epotch_time.seconds // 3600,
                                                                            epotch_time.seconds % 3600 // 60,
                                                                            epotch_time.seconds % 3600 % 60)
        print(str)
finally:
    if not para['debug'] and model.logger is not None and total_iter>5000:
        model.save(total_iter)
    # Clean up
    distributed.destroy_process_group()
time0.end_time()
print(time0.return_spent_time())
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 12446 --nproc_per_node=1 train.py --id s3_batch_4 --stage 3 --load_model saves/Apr20_13.41.23_s3_batch_4/Apr20_13.41.23_s3_batch_4_checkpoint.pth
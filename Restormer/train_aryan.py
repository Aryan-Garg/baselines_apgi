import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

from torch.utils.data import DataLoader
from einops import rearrange
from dataset_aryan import SPCDataset, SPCDataset_Mosaic

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            if dataset_opt["type"] == "mosaic":
                train_set = SPCDataset_Mosaic(file_list="/media/agarg54/Extreme SSD/dataset_files/combined_dataset.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            elif dataset_opt["type"] == "demosaic":
                train_set = SPCDataset(file_list="/media/agarg54/Extreme SSD/dataset_files/combined_dataset.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            else:
                raise NotImplementedError("Please specify either mosaic or demosaic option to be True.")
            train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=dataset_opt['batch_size_per_gpu'],
                    num_workers=24,
                    shuffle=True,
                    drop_last=True,
            )

            num_iter_per_epoch = len(train_set) // 2
            total_iters = int(opt['train']['total_iter'])
            total_epochs = 2
            
            print(f'Training statistics:\n\tNumber of train images: {len(train_set)}\n\tBatch size per gpu: 2\n\tWorld size (gpu number): {opt["world_size"]}\n\tRequire iter number per epoch: {num_iter_per_epoch}\n\tTotal epochs: {total_epochs} OR iters: {total_iters}.')

        elif phase == 'val':
            if dataset_opt["type"] == "mosaic":
                val_set = SPCDataset_Mosaic(file_list="/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            elif dataset_opt["type"] == "demosaic":
                val_set = SPCDataset(file_list="/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            else:
                raise NotImplementedError("Please specify either mosaic or demosaic option to be True.")
            val_loader = DataLoader(
                    dataset=val_set,
                    batch_size=dataset_opt['batch_size_per_gpu'],
                    num_workers=24,
                    shuffle=False,
                    drop_last=False,
            )
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)
    model = create_model(opt)
    train_loader, val_loader, total_epochs, total_iters = create_train_val_dataloader(opt, logger=None)

   
    # iters = opt['datasets']['train'].get('iters')
    # batch_size = 2
    # mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    # gt_size = opt['datasets']['train'].get('gt_size')
    # mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    # groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    # scale = opt['scale']

    for epoch in range(total_epochs):
        current_iter = 0
        for batch in tqdm(train_loader):
            gt, lq, _, _ = batch
            gt = rearrange((gt+1)/2, "b h w c -> b c h w").contiguous().float()
            lq = rearrange((lq+1)/2, "b h w c -> b c h w").contiguous().float()

            current_iter += 1
            if current_iter > total_iters:
                break
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            model.feed_train_data({'lq': lq, 'gt':gt})
            model.optimize_parameters(current_iter)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0 or current_iter == 1:
                print('Saving models and training states.')
                model.save(epoch, current_iter)
                for idx, batch in tqdm(enumerate(val_loader)):
                    gt, lq, _, _ = batch
                    gt = rearrange((gt+1)/2, "b h w c -> b c h w").contiguous().float()
                    lq = rearrange((lq+1)/2, "b h w c -> b c h w").contiguous().float()
                    model.feed_data({'lq': lq, 'gt':gt})
                    model.nonpad_test()
                    output_dict = model.get_current_visuals()
                    plt.imsave(f"./experiments/val_{opt['datasets']['val']['type']}/gt_{str(idx).zfill(4)}.png", rearrange(output_dict['gt'][0, ...], "c h w -> h w c").numpy())
                    plt.imsave(f"./experiments/val_{opt['datasets']['val']['type']}/lq_{str(idx).zfill(4)}.png", rearrange(output_dict['lq'][0, ...], "c h w -> h w c").numpy())
                    res = rearrange(output_dict['result'][0, ...], "c h w -> h w c")
                    res = (res - res.min()) / (res.max() - res.min())
                    plt.imsave(f"./experiments/val_{opt['datasets']['val']['type']}/out_{str(idx).zfill(4)}.png", res.numpy())

            # if current_iter % opt['logger']['save_img'] == 1:
            #     output_dict = model.get_current_visuals()
            #     os.makedirs("./experiments/temp", exist_ok=True)
            #     plt.imsave("./experiments/temp/gt.png", rearrange(output_dict['gt'][0, ...], "c h w -> h w c").numpy())
            #     plt.imsave("./experiments/temp/lq.png", rearrange(output_dict['lq'][0, ...], "c h w -> h w c").numpy())
            #     res = rearrange(output_dict['result'][0, ...], "c h w -> h w c")
            #     res = (res - res.min()) / (res.max() - res.min())
            #     plt.imsave("./experiments/temp/result.png", res.numpy())
            
            if current_iter > total_iters:
                break

        if current_iter > total_iters:
            break

    # end of epoch

    


if __name__ == '__main__':
    main()

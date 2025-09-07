import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
from os import path as osp

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)

from torch.utils.data import DataLoader
from dataset_aryan import SPCDataset, SPCDataset_Mosaic

from accelerate import Accelerator
from accelerate.utils import set_seed


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
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
            if opt["mosaic"]:
                train_set = SPCDataset_Mosaic(file_list="/mnt/disks/behemoth/datasets/dataset_txt_files/combined_dataset.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            elif opt["demosaic"]:
                train_set = SPCDataset(file_list="/mnt/disks/behemoth/datasets/dataset_txt_files/combined_dataset.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            else:
                raise NotImplementedError("Please specify either mosaic or demosaic option to be True.")
            train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=2,
                    num_workers=96,
                    shuffle=True,
                    drop_last=True,
            )

            num_iter_per_epoch = len(train_set) // 2
            total_iters = int(opt['train']['total_iter'])
            total_epochs = 1
            
            print(f'Training statistics:\n\tNumber of train images: {len(train_set)}\n\tBatch size per gpu: 2\n\tWorld size (gpu number): {opt["world_size"]}\n\tRequire iter number per epoch: {num_iter_per_epoch}\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            if opt["mosaic"]:
                val_set = SPCDataset_Mosaic(file_list="/mnt/disks/behemoth/datasets/dataset_txt_files/random_val.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            elif opt["demosaic"]:
                val_set = SPCDataset(file_list="/mnt/disks/behemoth/datasets/dataset_txt_files/random_val.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            else:
                raise NotImplementedError("Please specify either mosaic or demosaic option to be True.")
            val_loader = DataLoader(
                    dataset=val_set,
                    batch_size=2,
                    num_workers=96,
                    shuffle=False,
                    drop_last=True,
            )
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loader, total_epochs, total_iters


def main():
    opt = parse_options(is_train=True)
    ckpt = torch.load("/home/argar/apgi/baselines_apgi/NAFNet/pretrained_weights/NAFNet-SIDD-width64.pth")
    model = create_model(opt)
    try:
        model.load_state_dict(ckpt, strict=True)
    except Exception as e:
        print(e)
        print("\n[!] Loading pretrained model with strict=False")
        model.load_state_dict(ckpt, strict=False)
        
    train_loader, total_epochs, total_iters = create_train_val_dataloader(opt, logger=None)

    for epoch in range(total_epochs):
        current_iter = 0
        for batch in train_loader:
            gt, lq, prompt, gt_path = batch

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(lq, gt, is_val=False)
            result_code = model.optimize_parameters(current_iter)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                print('Saving models and training states.')
                model.save(epoch, current_iter)

    # end of epoch

    print('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    

if __name__ == '__main__':
    main()

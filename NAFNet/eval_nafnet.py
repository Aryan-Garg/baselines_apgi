import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
from os import path as osp
import copy

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import dict2str, parse
from basicsr.utils.dist_util import get_dist_info, init_dist

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
            if dataset_opt["simulation_type"] == "mosaic":
                train_set = SPCDataset_Mosaic(file_list="/mnt/disks/behemoth/datasets/dataset_txt_files/combined_dataset.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            elif dataset_opt["simulation_type"] == "demosaic":
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
            total_epochs = 2
            
            print(f'Training statistics:\n\tNumber of train images: {len(train_set)}\n\tBatch size per gpu: 2\n\tWorld size (gpu number): {opt["world_size"]}\n\tRequire iter number per epoch: {num_iter_per_epoch}\n\tTotal epochs: {total_epochs} OR iters: {total_iters}.')

        elif phase == 'val':
            if dataset_opt["simulation_type"] == "mosaic":
                val_set = SPCDataset_Mosaic(file_list="/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            elif dataset_opt["simulation_type"] == "demosaic":
                val_set = SPCDataset(file_list="/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            else:
                raise NotImplementedError("Please specify either mosaic or demosaic option to be True.")
            val_loader = DataLoader(
                    dataset=val_set,
                    batch_size=1,
                    num_workers=24,
                    shuffle=False,
                    drop_last=False,
            )
            total_epochs = 0
            total_iters = 0
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loader, total_epochs, total_iters


import piq
def compute_full_reference_metrics(gt_img, out_img):
    psnr = piq.psnr(out_img, gt_img, data_range=1., reduction='none')
    ssim = piq.ssim(out_img, gt_img, data_range=1.) 
    lpips = piq.LPIPS(reduction='none')(out_img, gt_img)
    return psnr.item(), ssim.item(), lpips.item()


import pyiqa
# from DeQAScore.src import Scorer
MODELmaniqa = pyiqa.create_metric('maniqa', device=torch.device("cuda"))
MODELclipiqa = pyiqa.create_metric('clipiqa', device=torch.device("cuda"))
MODELmusiq = pyiqa.create_metric('musiq', device=torch.device("cuda"))
def compute_no_reference_metrics(out_img):
    _, _, h, w = out_img.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    out_img = out_img[:, :, top:top+224, left:left+224]

    # ManIQA DeQA MUSIQ ClipIQA
    maniqa_score = MODELmaniqa(out_img).item()
    clipiqa_score = MODELclipiqa(out_img).item()
    musiq_score = MODELmusiq(out_img).item()

    return maniqa_score, clipiqa_score, musiq_score #, deqa_score


def single_image_inference(model, img, save_path, gt=None, calcMetrics=True):
    model.feed_data(lq=img, gt=gt, is_val=True)

    if model.opt['val'].get('grids', False):
        model.grids()
    model.test()
    if model.opt['val'].get('grids', False):
        model.grids_inverse()
    visuals = model.get_current_visuals()
    
    if "demosaic" in save_path:
        # print(img.size())
        input_img = img[:,[0],:,:].repeat(1,3,1,1)
        sr_img = visuals['result'][:,[0],:,:].clamp(0,1).repeat(1,3,1,1)
        gt_img = gt[:,[0],:,:].repeat(1,3,1,1)
    else:
        input_img = img
        sr_img = visuals['result'].clamp(0,1)
        gt_img = gt

    input_img = tensor2img(input_img)
    sr_img = tensor2img(sr_img)
    gt_img = tensor2img(gt_img)

    imwrite(sr_img, save_path)
    imwrite(input_img, f"{save_path[:-4]}_input.png")
    imwrite(gt_img, f"{save_path[:-4]}_gt.png")

    psnr, ssim, lpips, manIQA, clipIQA, musiq = None, None, None, None, None, None
    if calcMetrics:
        psnr, ssim, lpips = compute_full_reference_metrics(gt, visuals['result'].clamp(0,1))
        manIQA, clipIQA, musiq = compute_no_reference_metrics(visuals['result'].clamp(0,1))

    return psnr, ssim, lpips, manIQA, clipIQA, musiq



def main():
    opt = parse_options(is_train=False)
    model = create_model(opt)
    train_loader, val_loader, total_epochs, total_iters = create_train_val_dataloader(opt, logger=None)

    current_iter = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    avg_maniqa = 0
    avg_clipiqa = 0
    avg_musiq = 0
    os.makedirs(f"./evaluation_full_ds_{opt['datasets']['val']['simulation_type']}", exist_ok=True)
    # print("Len val loader:", len(val_loader))
    for batch in tqdm(val_loader):
        gt, lq, _, _ = batch
        gt = rearrange((gt+1)/2, "b h w c -> b c h w").contiguous().float()
        lq = rearrange((lq+1)/2, "b h w c -> b c h w").contiguous().float()
        current_iter += 1

        # eval
        psnr, ssim, lpips, manIQA, clipIQA, musiq = single_image_inference(model, 
                                                                           lq, 
                                                                           f"./evaluation_full_ds_{opt['datasets']['val']['simulation_type']}/{str(current_iter).zfill(4)}.png", 
                                                                           gt=gt,
                                                                           calcMetrics=True)

        if psnr is not None:
            avg_psnr += psnr
            avg_ssim += ssim
            avg_lpips += lpips
            avg_maniqa += manIQA
            avg_clipiqa += clipIQA
            avg_musiq += musiq
    
    N = len(val_loader)
    avg_psnr    /= N
    avg_ssim    /= N
    avg_lpips   /= N
    avg_maniqa  /= N
    avg_clipiqa /= N
    avg_musiq   /= N
    print(f"Avg. PSNR: {avg_psnr}")
    print(f"Avg. SSIM: {avg_ssim}")
    print(f"Avg. LPIPS: {avg_lpips}")
    print(f"Avg. ManIQA: {avg_maniqa}")
    print(f"Avg. ClipIQA: {avg_clipiqa}")
    print(f"Avg. MUSIQ: {avg_musiq}")


if __name__ == '__main__':
    main()

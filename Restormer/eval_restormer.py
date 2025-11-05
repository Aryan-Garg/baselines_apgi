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
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse
from basicsr.utils import get_root_logger, imwrite, tensor2img

from torch.utils.data import DataLoader
from einops import rearrange
from dataset_aryan import SPCDataset, SPCDataset_Mosaic

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def parse_options(is_train=False):
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
            if dataset_opt["type"] == "mosaic":
                train_set = SPCDataset_Mosaic(file_list="/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            elif dataset_opt["type"] == "demosaic":
                train_set = SPCDataset(file_list="/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", 
                                              out_size=256, crop_type="center", use_hflip=False, bits=3)
            else:
                raise NotImplementedError("Please specify either mosaic or demosaic option to be True.")
            train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=dataset_opt['batch_size_per_gpu'],
                    num_workers=96,
                    shuffle=True,
                    drop_last=True,
            )

            num_iter_per_epoch = len(train_set) // 2
            total_iters = int(opt['train']['total_iter'])
            total_epochs = 2
            
            print(f'Training statistics:\n\tNumber of train images: {len(train_set)}\n\tBatch size per gpu: 2\n\tWorld size (gpu number): {opt["world_size"]}\n\tRequire iter number per epoch: {num_iter_per_epoch}\n\tTotal epochs: {total_epochs} OR iters: {total_iters}.')

        elif phase == 'val':
            total_epochs, total_iters = 0, 0
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


import piq
def compute_full_reference_metrics(gt_img, out_img):
    psnr = piq.psnr(out_img, gt_img, data_range=1., reduction='none')
    ssim = piq.ssim(out_img, gt_img, data_range=1.) 
    lpips = piq.LPIPS(reduction='none')(out_img, gt_img)
    return psnr.item(), ssim.item(), lpips.item()


import pyiqa
# from DeQAScore.src import Scorer
# ManIQA DeQA MUSIQ ClipIQA
maniqaMODEL  = pyiqa.create_metric('maniqa', device=torch.device("cuda"))
clipiqaMODEL = pyiqa.create_metric('clipiqa', device=torch.device("cuda"))
musiqMODEL   = pyiqa.create_metric('musiq', device=torch.device("cuda"))
def compute_no_reference_metrics(out_img):
    _, _, h, w = out_img.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    out_img = out_img[:, :, top:top+224, left:left+224]

    maniqa_score = maniqaMODEL(out_img).item()
    clipiqa_score = clipiqaMODEL(out_img).item()
    musiq_score = musiqMODEL(out_img).item()

    return maniqa_score, clipiqa_score, musiq_score #, deqa_score


def calcMetrics(gt, out):
    psnr, ssim, lpips, manIQA, clipIQA, musiq = None, None, None, None, None, None
    psnr, ssim, lpips = compute_full_reference_metrics(gt, out)
    manIQA, clipIQA, musiq = compute_no_reference_metrics(out)
    return psnr, ssim, lpips, manIQA, clipIQA, musiq


def main():
    # parse options, set distributed setting, set ramdom seed
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
    for batch in tqdm(val_loader):
        gt, lq, _, _ = batch
        gt = rearrange((gt+1)/2, "b h w c -> b c h w").contiguous().float()
        lq = rearrange((lq+1)/2, "b h w c -> b c h w").contiguous().float()
        data = {'lq': lq, 'gt':gt}

        model.feed_data(data)
        model.nonpad_test()

        out = model.output
        # print(out.shape, out.min(), out.max())
        # print(gt.shape, gt.min(), gt.max())
        # exit()
        psnr, ssim, lpips, manIQA, clipIQA, musiq = calcMetrics(gt.to(torch.device('cuda')), out.clamp(0,1))
        if psnr is not None:
            avg_psnr += psnr
            avg_ssim += ssim
            avg_lpips += lpips
            avg_maniqa += manIQA
            avg_clipiqa += clipIQA
            avg_musiq += musiq

        inp_img = tensor2img(lq, rgb2bgr=True)
        sr_img = tensor2img(out, rgb2bgr=True)
        gt_img = tensor2img(gt, rgb2bgr=True)

        save_path = f"/media/agarg54/Extreme SSD/code/baselines_apgi/Restormer/experiments/val_{opt['datasets']['val']['type']}/{str(current_iter).zfill(4)}.png"
        imwrite(sr_img, save_path)
        imwrite(inp_img, f"{save_path[:-4]}_lq.png")
        imwrite(gt_img, f"{save_path[:-4]}_gt.png")
        current_iter += 1
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

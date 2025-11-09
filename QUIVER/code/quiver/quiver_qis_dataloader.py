import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
import random
from random import choices
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import glob
import cv2
import quiver_qis_input_args
import qis_utils

class train_dataloader(Dataset):
    def __init__(self, args):
        super(train_dataloader, self).__init__()
        self.video_paths = sorted(glob.glob(os.path.join(args.gtdata_dir, '*.mp4')))
        self.args = args
        
    def __getitem__(self, idx):
        vid_path = self.video_paths[idx]
        gt_seq = qis_utils.frames_extraction(vid_path, self.args.num_frames, start_frame = None, downsample=self.args.downsample)
        gt_seq = self.transforms(gt_seq)

        qis_seq = qis_utils.sensor_image_simulation(self.args.avg_PPP, gt_seq, self.args.QE,
                                                         self.args.theta_dark, self.args.sigma_read,
                                                         self.args.clicks_per_frame, self.args.Nbits, self.args.gain)
        
        qis_seq = qis_seq[:, None, :, :]
        gt_seq = gt_seq[:, None, :, :]
        
        qis_seq, gt_seq = qis_utils.normalize(qis_seq, max_value=(2**self.args.Nbits) - 1), qis_utils.normalize(gt_seq,
                                                                                                               max_value=255.)
        qis_seq = torch.from_numpy(qis_seq)
        gt_seq = torch.from_numpy(gt_seq)

        return qis_seq, gt_seq

    def __len__(self):
        return len(self.video_paths)

    def transforms(self, gt_seq):
        if self.args.transforms:
            left = random.randint(0, gt_seq.shape[2] - self.args.patch_size)
            right = left + self.args.patch_size
            top = random.randint(0, gt_seq.shape[1] - self.args.patch_size)
            bottom = top + self.args.patch_size
            gt_seq = gt_seq[:, top:bottom, left:right]

            do_nothing = lambda x: x
            flipud = lambda x: x[::-1, :]
            rot90 = lambda x: np.rot90(x, axes=(0, 1))
            rot90_flipud = lambda x: (np.rot90(x, axes=(0, 1)))[::-1, :]
            rot180 = lambda x: np.rot90(x, k=2, axes=(0, 1))
            rot180_flipud = lambda x: (np.rot90(x, k=2, axes=(0, 1)))[::-1, :]
            rot270 = lambda x: np.rot90(x, k=3, axes=(0, 1))
            rot270_flipud = lambda x: (np.rot90(x, k=3, axes=(0, 1)))[::-1, :]

            N, _, _ = gt_seq.shape

            # define transformations and their frequency, then pick one.
            # 	aug_list = [do_nothing, scale_img, flipud, rot90, rot90_flipud, \
            # 				rot180, rot180_flipud, rot270, rot270_flipud]
            # 	w_aug = [7, 21, 2, 2, 2, 2, 2, 2, 2]
            aug_list = [do_nothing, flipud, rot90, rot90_flipud, rot180, rot180_flipud, rot270, rot270_flipud]
            w_aug = [7, 4, 4, 4, 4, 4, 4, 4]
            transf = choices(aug_list, w_aug)

            # transform all images in array
            for j in range(N):
                gt_seq[j, ...] = transf[0](gt_seq[j, ...])
        else:
            pass

        return gt_seq


class val_dataloader(Dataset):
    def __init__(self, args):
        super(val_dataloader, self).__init__()
        self.video_paths = sorted(glob.glob(os.path.join(args.valgtdata_dir, '*.mp4')))
        self.args = args
        #self.val_frames = args.val_num_frames

    def __getitem__(self, idx):
        vid_path = self.video_paths[idx]
        gt_seq = qis_utils.frames_extraction(vid_path, self.args.num_frames, start_frame = 50, downsample=self.args.downsample)
        #gt_seq = self.transforms(gt_seq)

        qis_seq = qis_utils.sensor_image_simulation(self.args.avg_PPP, gt_seq, self.args.QE,
                                                         self.args.theta_dark, self.args.sigma_read,
                                                         self.args.clicks_per_frame, self.args.Nbits, self.args.gain)
        qis_seq = qis_seq[:, None, :, :]
        gt_seq = gt_seq[:, None, :, :]

        qis_seq, gt_seq = qis_utils.normalize(qis_seq, max_value=(2**self.args.Nbits) - 1), qis_utils.normalize(gt_seq,
                                                                                                               max_value=255.)
        qis_seq = torch.from_numpy(qis_seq)
        gt_seq = torch.from_numpy(gt_seq)

        return qis_seq, gt_seq

    def __len__(self):
        return len(self.video_paths)

    def transforms(self, gt_seq):
        if self.args.transforms:
            left = 30
            right = left + self.args.patch_size
            top = 30
            bottom = top + self.args.patch_size
            gt_seq = gt_seq[:, top:bottom, left:right]
        else:
            pass

        return gt_seq


def chunk_list(args, lst):
    """
    Divide a list into chunks of a given size.
    """
    return [lst[i-args.past_frames:i + args.future_frames + 1] for i in range(args.past_frames, len(lst) - args.future_frames)]


class test_dataloader(Dataset):
    def __init__(self, args, video_path):
        super(test_dataloader, self).__init__()
        self.video_path = video_path
        self.args = args

    def __getitem__(self, idx):
        gt_seq = qis_utils.frames_extraction(self.video_path, self.args.num_frames, start_frame=idx, downsample=self.args.downsample)
        qis_seq = qis_utils.sensor_image_simulation(self.args.avg_PPP, gt_seq, self.args.QE,
                                                        self.args.theta_dark, self.args.sigma_read,
                                                        self.args.clicks_per_frame, self.args.Nbits, self.args.gain)
        qis_seq = qis_seq[:, None, :, :]
        gt_seq = gt_seq[:, None, :, :]

        qis_seq, gt_seq = qis_utils.normalize(qis_seq, max_value=(2 ** self.args.Nbits) - 1), \
            qis_utils.normalize(gt_seq, max_value=255.)
        qis_seq = torch.from_numpy(qis_seq)
        gt_seq = torch.from_numpy(gt_seq)

        return qis_seq, gt_seq

    def __len__(self):
        vid = cv2.VideoCapture(self.video_path)
        return int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - self.args.num_frames + 1
        

def load_file_list(file_list_path: str):
    files = []
    with open(file_list_path, "r") as fin:
        for line in fin:
            p = line.strip()
            if p:
                p = p.split(" ")
                path = p[0]
                prompt = ""
                if len(p) > 1:
                    prompt = " ".join(p[1:])
                files.append({"image_path": path, "prompt": prompt})
    return files

from PIL import Image
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def srgb_to_linearrgb(img):
    """Performs sRGB to linear RGB color space conversion by reversing gamma
    correction and obtaining values that represent the scene's intensities.

    Args:
        img: npy array or torch tensor

    Returns:
        linear rgb image.
    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.04045
    img[mask] = module.clip(img[mask], 0.0, module.inf) / 12.92
    img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4  # type: ignore
    return img

from typing import Sequence, Dict, Union, List, Mapping, Any, Optional, cast
import numpy.typing as npt
def emulate_spc(
        img, factor: float = 1.0):
    # Perform bernoulli sampling (equivalent to binomial w/ n=1)
    rng = np.random.default_rng()
    # print("inside rng:", 1.0 - np.exp(-img * factor))
    return rng.binomial(cast(npt.NDArray[np.integer], 1), 1.0 - np.exp(-img * factor))


class spadtest_dataloader(Dataset):
    def __init__(self, args, file_list_path="/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", bits=3, out_size=512):
        super(spadtest_dataloader, self).__init__()
        self.img_list = load_file_list(file_list_path=file_list_path)
        self.args = args
        self.HARDDISK_DIR = "/media/agarg54/Extreme SSD/"
        self.bits = bits
        self.out_size = out_size
        print(f"[+] Sim bits = {self.bits}")


    def generate_spc_from_gt(self, img_gt, N=1):
        if img_gt is None:
            return None
        img = srgb_to_linearrgb(img_gt / 255.)
        img = emulate_spc(img, 
                          factor= 1. / N # Brightness directly proportional to this hparam. 1.0 => scene's natural lighting
                        )
        return img


    def __getitem__(self, idx):
        # print(f"Opening: {self.HARDDISK_DIR + self.img_list[idx]['image_path'][2:]}")
        gt_img = qis_utils.open_image(self.HARDDISK_DIR + self.img_list[idx]['image_path'][2:], 
                                      gray_mode=True, expand_if_needed=False, 
                                      expand_axis0=False, normalize_data=False)[0]
        # print(type(gt_img), gt_img.shape, gt_img.dtype)
        gt_img = center_crop_arr(Image.fromarray(gt_img), self.out_size)
        # img_lq_sum = np.zeros_like(gt_img, dtype=np.float32)
        # NOTE: No motion-blur. Assumes SPC-fps >>> scene motion
        # N = 2**self.bits - 1
        # for i in range(N): # 3-bit (2**3 - 1)
        #     img_lq_sum = img_lq_sum + self.generate_spc_from_gt(gt_img)
        # img_lq = img_lq_sum / (1.0*N)
        seq_list = []
        # qis_seq = []
        for i in range(11):
            seq_list.append(gt_img)
            # qis_seq.append(img_lq)
        gt_seq = np.stack(seq_list, axis=0)
        # qis_seq = np.stack(qis_seq, axis=0)
        qis_seq = qis_utils.sensor_image_simulation(self.args.avg_PPP, gt_seq, self.args.QE,
                                                        self.args.theta_dark, self.args.sigma_read,
                                                        self.args.clicks_per_frame, self.args.Nbits, self.args.gain)
        
        qis_seq = qis_seq[:, None, :, :]
        gt_seq = gt_seq[:, None, :, :]

        qis_seq, gt_seq = qis_utils.normalize(qis_seq, max_value=(2 ** self.args.Nbits) - 1), \
            qis_utils.normalize(gt_seq, max_value=255.)
        
        qis_seq = torch.from_numpy(qis_seq)
        gt_seq = torch.from_numpy(gt_seq)

        return qis_seq, gt_seq

    def __len__(self):
        return len(self.img_list)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='qis reconstruction dataloader args')
    quiver_qis_input_args.rtmb_training_args(parser)
    quiver_qis_input_args.sensor_args(parser)
    args = parser.parse_args()

    trainset = train_dataloader(args)
    train_dataloader = DataLoader(dataset=trainset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    qis_seq, gt_seq = next(iter(train_dataloader))
    print('')

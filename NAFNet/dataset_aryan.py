from typing import Sequence, Dict, Union, List, Mapping, Any, Optional, cast
import math
import time
import io
import random
import importlib

import numpy as np
import numpy.typing as npt
import cv2
from PIL import Image
import polars as pl
import torch
from torch.nn import functional as F
import torch.utils.data as data
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

from utils.common import instantiate_from_config


def load_file_list(file_list_path: str) -> List[Dict[str, str]]:
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


def load_video_file_list(file_list_path: str) -> List[Dict[str, str]]:
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
                files.append({"video_path": path, "prompt": prompt})
    return files


def load_file_metas(file_metas: List[Dict[str, str]]) -> List[Dict[str, str]]:
    files = []
    for file_meta in file_metas:
        file_list_path = file_meta["file_list"]
        image_path_key = file_meta["image_path_key"]
        short_prompt_key = file_meta["short_prompt_key"]
        long_prompt_key = file_meta["long_prompt_key"]
        ext = os.path.splitext(file_list_path)[1].lower()
        assert ext == ".parquet", f"only support parquet format"
        df = pl.read_parquet(file_list_path)
        for row in df.iter_rows(named=True):
            files.append(
                {
                    "image_path": row[image_path_key],
                    "short_prompt": row[short_prompt_key],
                    "long_prompt": row[long_prompt_key],
                }
            )
    return files


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
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


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        # img: torch.Tensor
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


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


def emulate_spc(
        img, factor: float = 1.0) -> npt.NDArray[np.integer]:
    """Perform bernoulli sampling on linearized RGB frames to yield binary frames.

    Args:
        img (npt.ArrayLike): Linear intensity image to sample binary frame from.
        factor (float, optional): Arbitrary corrective brightness factor. Defaults to 1.0.
        rng (np.random.Generator, optional): Random number generator. Defaults to None.

    Returns:
        Binary single photon frame in [0,1]

        Up --> More PPP
        down --> Low-light

        1    -> 0.3           PPP
        1000 -> 0.3/1000  --> V. Dark
    """
    # Perform bernoulli sampling (equivalent to binomial w/ n=1)
    rng = np.random.default_rng()
    # print("inside rng:", 1.0 - np.exp(-img * factor))
    return rng.binomial(cast(npt.NDArray[np.integer], 1), 1.0 - np.exp(-img * factor))


def mle_intensity_from_S(S, n_q, tau=1.0, eta=1.0, fix_inf=True):
    """MLE estimate phi = -ln(1 - S/n_q) / (tau * eta). S can be int array."""
    S = np.asarray(S, dtype=float)
    if fix_inf:
        # avoid S == n_q (perfect saturation) -> infinite lambda
        S = np.where(S >= n_q, n_q - 1.0, S)
    p_hat = S / float(n_q)
    # numerical safety: clip p_hat to [0, 1-ε]
    eps = 1e-12
    p_hat = np.clip(p_hat, 0.0, 1.0 - eps)
    phi = -np.log(1.0 - p_hat) / (tau * eta)
    return phi


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get(self, filepath: str) -> bytes:
        pass


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf
    

class SPCDataset_Mosaic(data.Dataset):
    """
        Dataset for finetuning the VAE's encoder and Adversarial FT Stages (independent of each other).
        Args:
            file_list (str): Path to the file list containing image paths and prompts.
            file_backend_cfg (Mapping[str, Any]): Configuration for the file backend to load images.
            out_size (int): The output size of the images after cropping.
            crop_type (str): Type of cropping to apply to the images. Options are 'none', 'center', or 'random'.
        Returns:
            A dictionary containing:
                - 'gt': Ground truth image tensor of shape (C, H, W) with pixel values in the range [-1, 1].
                - 'lq': SPC image (dubbed as low-quality) tensor of shape (C, H, W) with pixel values in the range [-1, 1].
                - 'prompt': The prompt associated with the image.
    """
    def __init__(self,
                    file_list: str,
                    out_size: int,
                    crop_type: str,
                    use_hflip: bool,
                    bits=3) -> "SPCDataset_Mosaic":

        super(SPCDataset_Mosaic, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = HardDiskBackend()
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip # No need for 1.5M big dataset
        assert self.crop_type in ["none", "center", "random"]
        self.HARDDISK_DIR = "/media/agarg54/Extreme SSD/"
        self.bits = bits
        print(f"[+] Sim bits = {self.bits}")


    def get_mosaic(self, img):
        """
            Convert a demosaiced RGB image (HxWx3) into an RGGB Bayer mosaic.
        """
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        bayer = np.zeros_like(img)

        bayer_pattern_type = random.choice(["RGGB", "GRBG", "BGGR", "GBRG"])

        if bayer_pattern_type == "RGGB":
            # Red
            bayer[0::2, 0::2, 0] = R[0::2, 0::2]
            # Green
            bayer[0::2, 1::2, 1] = G[0::2, 1::2]
            bayer[1::2, 0::2, 1] = G[1::2, 0::2]
            # Blue
            bayer[1::2, 1::2, 2] = B[1::2, 1::2]
        elif bayer_pattern_type == "GRBG":
            # Red
            bayer[0::2, 1::2, 0] = R[0::2, 1::2]
            # Green 
            bayer[0::2, 0::2, 1] = G[0::2, 0::2]
            bayer[1::2, 1::2, 1] = G[1::2, 1::2]
            # Blue
            bayer[1::2, 0::2, 2] = B[1::2, 0::2]
            
        elif bayer_pattern_type == "BGGR":
            # Blue
            bayer[0::2, 0::2, 2] = B[0::2, 0::2]
            # Green
            bayer[0::2, 1::2, 1] = G[0::2, 1::2]
            bayer[1::2, 0::2, 1] = G[1::2, 0::2]
            # Red
            bayer[1::2, 1::2, 0] = R[1::2, 1::2]
        
        else: # GBRG
            # Green
            bayer[0::2, 0::2, 1] = G[0::2, 0::2]
            bayer[1::2, 1::2, 1] = G[1::2, 1::2]
            # Blue
            bayer[0::2, 1::2, 2] = B[0::2, 1::2]
            # Red
            bayer[1::2, 0::2, 0] = R[1::2, 0::2]

        return bayer


    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image


    def generate_spc_from_gt(self, img_gt, N=1):
        if img_gt is None:
            return None
        img = srgb_to_linearrgb(img_gt / 255.)
        img = emulate_spc(img, 
                          factor= 1. / N # Brightness directly proportional to this hparam. 1.0 => scene's natural lighting
                        )
        return img


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        img_lq = None
        while img_gt is None and img_lq is None:
            # load meta file
            img_path = self.image_files[index]['image_path']
            gt_path =  self.HARDDISK_DIR + img_path[2:]
            # print("gt path:", gt_path)
            # print(f"Loading GT image from {gt_path}")
            prompt = self.image_files[index]['prompt']

            try:
                img_gt = self.load_gt_image(gt_path)
            except Exception as e:
                print(e)
                print(f"Could not load: {gt_path}, setting a random index")
                index = random.randint(0, len(self) - 1)
                continue
            
            if img_gt is None:
                print(f"failed to load {gt_path} or generate lq image, try another image")
                index = random.randint(0, len(self) - 1)
                continue


            img_lq_sum = np.zeros_like(img_gt, dtype=np.float32)
            # NOTE: No motion-blur. Assumes SPC-fps >>> scene motion
            N = 2**self.bits - 1
            for i in range(N): # 3-bit (2**3 - 1)
                img_lq_sum = img_lq_sum + self.get_mosaic(self.generate_spc_from_gt(img_gt))
            img_lq = img_lq_sum / (1.0*N)


        # Shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_gt = (img_gt / 255.0).astype(np.float32)
        img_lq = img_lq.astype(np.float32) # BUG-FIXED now!!! for all datasets img_lq is already [0,1], no need to divide by 255


        # Should lq be normalized to [-1,1] or stay in [0, 1] range? For now [-1, 1]
        gt = (img_gt * 2 - 1).astype(np.float32)
        # [-1, 1]
        lq = (img_lq * 2 - 1).astype(np.float32) 
        # print(np.amax(lq), np.amin(lq))
        return gt, lq, prompt, gt_path


    def __len__(self) -> int:
        return len(self.image_files)
    

class SPCDataset(data.Dataset):
    """
        Dataset for finetuning the VAE's encoder and Adversarial FT Stages (independent of each other).
        Args:
            file_list (str): Path to the file list containing image paths and prompts.
            file_backend_cfg (Mapping[str, Any]): Configuration for the file backend to load images.
            out_size (int): The output size of the images after cropping.
            crop_type (str): Type of cropping to apply to the images. Options are 'none', 'center', or 'random'.
        Returns:
            A dictionary containing:
                - 'gt': Ground truth image tensor of shape (C, H, W) with pixel values in the range [-1, 1].
                - 'lq': SPC image (dubbed as low-quality) tensor of shape (C, H, W) with pixel values in the range [-1, 1].
                - 'prompt': The prompt associated with the image.
    """
    def __init__(self,
                    file_list: str,
                    out_size: int,
                    crop_type: str,
                    use_hflip: bool,
                    bits=3) -> "SPCDataset":

        super(SPCDataset, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = HardDiskBackend()
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip # No need for 1.5M big dataset
        assert self.crop_type in ["none", "center", "random"]
        self.HARDDISK_DIR = "/media/agarg54/Extreme SSD/"
        self.bits = bits
        print(f"[+] Sim bits = {self.bits}")


    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image


    def generate_spc_from_gt(self, img_gt, N=1):
        if img_gt is None:
            return None
        img = srgb_to_linearrgb(img_gt / 255.)
        img = emulate_spc(img, 
                          factor= 1. / N # Brightness directly proportional to this hparam. 1.0 => scene's natural lighting
                        )
        return img


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        img_lq = None
        while img_gt is None and img_lq is None:
            # load meta file
            img_path = self.image_files[index]['image_path']
            gt_path =  self.HARDDISK_DIR + img_path[2:]
            # print("gt path:", gt_path)
            # print(f"Loading GT image from {gt_path}")
            prompt = self.image_files[index]['prompt']

            try:
                img_gt = self.load_gt_image(gt_path)
            except Exception as e:
                print(e)
                print(f"Could not load: {gt_path}, setting a random index")
                index = random.randint(0, len(self) - 1)
                continue

            if img_gt is None:
                print(f"failed to load {gt_path} or generate lq image, try another image")
                index = random.randint(0, len(self) - 1)
                continue


            img_lq_sum = np.zeros_like(img_gt, dtype=np.float32)
            # NOTE: No motion-blur. Assumes SPC-fps >>> scene motion
            N = 2**self.bits - 1
            for i in range(N): # 3-bit (2**3 - 1)
                img_lq_sum = img_lq_sum + self.generate_spc_from_gt(img_gt)
            img_lq = img_lq_sum / (1.0*N)


        # Shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_gt = (img_gt / 255.0).astype(np.float32)
        img_lq = img_lq.astype(np.float32) 

        # Should lq be normalized to [-1,1] or stay in [0, 1] range? For now [-1, 1]
        gt = (img_gt * 2 - 1).astype(np.float32)
        # [-1, 1]
        lq = (img_lq * 2 - 1).astype(np.float32) 
        # print(np.amax(lq), np.amin(lq))
        return gt, lq, prompt, gt_path


    def __len__(self) -> int:
        return len(self.image_files)
    

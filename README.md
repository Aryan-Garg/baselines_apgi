# Baselines for CVPR 2026 Paper: gQIR

### 1. Instant IR

Was not finetuned.

@article{huang2024instantir,
  title={InstantIR: Blind Image Restoration with Instant Generative Reference},
  author={Huang, Jen-Yuan and Wang, Haofan and Wang, Qixun and Bai, Xu and Ai, Hao and Xing, Peng and Huang, Jen-Tse},
  journal={arXiv preprint arXiv:2410.06551},
  year={2024}
}

### 2. Restormer

> CUDA_VISIBLE_DEVICES=13 python3 train_aryan.py -opt Denoising/Options/FT_demosaic_real.yml
> CUDA_VISIBLE_DEVICES=12 python3 train_aryan.py -opt Denoising/Options/FT_mosaic_real.yml

### 3. NAFNet

> CUDA_VISIBLE_DEVICES=15 python3 train_aryan.py -opt options/train/FT_demosaic_width64.yml
> CUDA_VISIBLE_DEVICES=14 python3 train_aryan.py -opt options/train/FT_mosaic_width64.yml

### 4. QBP 

### 5. QUIVER

### 6*. bit2bit

Future work baseline



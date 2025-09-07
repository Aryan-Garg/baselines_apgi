# Baselines for CVPR 2026 Paper: gQIR

### 1. Instant IR

Was not finetuned since it claims 'any' degradation fixing capabilities.

```bibtex
@article{huang2024instantir,
  title={InstantIR: Blind Image Restoration with Instant Generative Reference},
  author={Huang, Jen-Yuan and Wang, Haofan and Wang, Qixun and Bai, Xu and Ai, Hao and Xing, Peng and Huang, Jen-Tse},
  journal={arXiv preprint arXiv:2410.06551},
  year={2024}
}
```

### 2. Restormer

> CUDA_VISIBLE_DEVICES=13 python3 train_aryan.py -opt Denoising/Options/FT_demosaic_real.yml
> CUDA_VISIBLE_DEVICES=12 python3 train_aryan.py -opt Denoising/Options/FT_mosaic_real.yml

```bibtex
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```

### 3. NAFNet

> CUDA_VISIBLE_DEVICES=15 python3 train_aryan.py -opt options/train/FT_demosaic_width64.yml
> CUDA_VISIBLE_DEVICES=14 python3 train_aryan.py -opt options/train/FT_mosaic_width64.yml

```bibtex
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
```
### 4. QBP 

```bibtex
@article{ma_quanta_2020,
    title = “Quanta Burst Photography”,
    author = “Ma, Sizhuo and Gupta, Shantanu and Ulku, Arin C. and Brushini, Claudio and Charbon, Edoardo and Gupta, Mohit”,
    journal = “ACM Transactions on Graphics (TOG)”,
    doi = “10.1145/3386569.3392470”,
    volume = “39”,
    number = “4”,
    year = “2020”,
    month = “7”
    publisher = “ACM”
}
```

### 5. QUIVER

```bibtex
@inproceedings{chennuri2024quanta,
  title={Quanta Video Restoration},
  author={Chennuri, Prateek and Chi, Yiheng and Jiang, Enze and Godaliyadda, GM Dilshan and Gnanasambandam, Abhiram and Sheikh, Hamid R and Gyongy, Istvan and Chan, Stanley H},
  booktitle={European Conference on Computer Vision},
  pages={152--171},
  year={2024},
  organization={Springer}
}
```

### 6*. bit2bit

Future work baseline



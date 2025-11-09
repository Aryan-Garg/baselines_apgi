#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 eval_quiver.py \
 --weights_path "/media/agarg54/Extreme SSD/code/baselines_apgi/QUIVER/code/quiver/weights/quiver_best_p5f5_3.25PPP.pth" \
 --n_features 64 --avg_PPP 3.25 --sigma_read 0
#!/bin/bash

python quiver_qis_test.py \
 --weights_path weights/quiver_best_p5f5_3.25PPP.pth --n_features 64 \
 --testgtdata_dir /home/argar/datasets/japan-alley/interp-travel/quiver_ingestible  -avg_PPP 3.25
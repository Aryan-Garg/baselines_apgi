#!/bin/bash

python quiver_qis_test.py \
 --weights_path weights/quiver_best_p5f5_26PPP.pth --n_features 64 -avg_PPP 26 \
 --testgtdata_dir /nobackup3/aryan/apgi/datasets/i2-2kfps_v1/special_test/
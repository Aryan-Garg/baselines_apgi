#!/bin/bash

python quiver_qis_test.py \
 --weights_path weights/quiver_best_p5f5_3.25PPP.pth --n_features 64 \
 --testgtdata_dir /nobackup3/aryan/apgi/datasets/tog_moped/test --downsample 1
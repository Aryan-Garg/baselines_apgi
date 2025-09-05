### Self-Supervised U-Net for denosing

Run example.ipynb to train
config.yml to change hparams
and example_inference.ipynb for inference stuff

~6hrs (10_000 x 128 x 128)

---   

## TODO: Make visionsim dataset for this similar to the /mnt/disks/behemoth/datasets/bit2bit_data_scaling_data.h5 & /mnt/disks/behemoth/datasets/bit2bit_leave_out_test.h5

Hypothesis: 14.7M params is too low to represent many scenes? 

Bit2bit --> Train & test on leave out test seq.
Bit2bit --> Train on 10seq --> test within training sequences.
Bit2bit --> Train on 10seq --> test on out of distribution.

---   

### NOTE
Prior-free MLP Reconstruction:

in: TxHxW 
out:TxHxW
MSE reconstruction

---
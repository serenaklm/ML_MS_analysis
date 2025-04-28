#!/bin/bash

# ratios=(0.05 0.1 0.1)

for ratio in 0.10 0.30 0.50 0.70 0.90;
do

    echo "Training with sampling ratio = $ratio"
    python train.py --config_file "w_meta_config.yaml" --sampling_ratio "$ratio" --random

done
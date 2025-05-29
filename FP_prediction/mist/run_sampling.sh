#!/bin/bash

# ratios=(0.05 0.1 0.1)

for ratio in 0.90 0.85 0.80 0.75 0.70 0.50 0.30 0.10 0.05 0.01;
do

    echo "Training with sampling ratio = $ratio"
    python train.py --config_file "w_meta_config_2.yaml" --sampling_ratio "$ratio" --strategy "remove_top_k_harmful"

done
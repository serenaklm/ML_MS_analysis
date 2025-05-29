#!/bin/bash

for ratio in 0.80 0.85;

do

    echo "Training with sampling ratio = $ratio"
    python train.py --config_file "w_meta_config.yaml" --sampling_ratio "$ratio" --random

done

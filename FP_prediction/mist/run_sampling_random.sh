#!/bin/bash

# ratios=(0.05 0.1 0.1)

for ratio in 0.10 0.20 0.30 0.40 0.50;
do

    echo "Running with sampling ratio = $ratio"
    python train.py --sampling_ratio "$ratio" --random

done
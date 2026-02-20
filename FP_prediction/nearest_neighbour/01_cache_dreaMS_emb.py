"""
    Compute and cache DreaMS embeddings for train/test MGF splits.

    This script iterates over a set of datasets and split definitions, loads
    `train.mgf` and `test.mgf` files from a cache directory, computes DreaMS
    embeddings using `dreams.api.dreams_embeddings`, and saves the resulting
    embeddings as pickled arrays.

    This API call uses the default ssl_model.ckpt checkpoint without contrastive fine-tuning,
    to avoid potential data leakage in our evaluations.

"""

import os
import numpy as np
from pathlib import Path
from dreams.api import dreams_embeddings

from utils import pickle_data

if __name__ == "__main__":

    MGF_cache_folder = Path("../data/cache/MGF_files")
    cache_folder = Path("../cache/DreaMS_emb")
    if not os.path.exists(cache_folder): os.makedirs(cache_folder)

    datasets = ["canopus", "massspecgym", "nist2023"]
    splits = ["scaffold_vanilla", "inchikey_vanilla", "random", "LS"]

    for dataset in datasets:

        for split in splits: 

            emb_folder = os.path.join(cache_folder, dataset, split)
            if not os.path.exists(emb_folder): os.makedirs(emb_folder)

            train_emb_path = os.path.join(emb_folder, "train.pkl")

            if not os.path.exists(train_emb_path):
                    
                train_MGF_path = os.path.join(MGF_cache_folder, dataset, split, "train.mgf")
                emb = dreams_embeddings(train_MGF_path)
                pickle_data(emb, train_emb_path)
                print(f"Gotten embeddings for {dataset}, {split} (train), {emb.shape}")

            test_emb_path = os.path.join(emb_folder, "test.pkl")
            if not os.path.exists(test_emb_path):
                
                test_MGF_path = os.path.join(MGF_cache_folder, dataset, split, "test.mgf")
                emb = dreams_embeddings(test_MGF_path)
                pickle_data(emb, test_emb_path)
                print(f"Gotten embeddings for {dataset}, {split} (test), {emb.shape}")
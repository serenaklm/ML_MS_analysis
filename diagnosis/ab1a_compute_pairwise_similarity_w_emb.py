import os 
import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_pickle, load_json, pickle_data

def compute_topk_similarities(test_MS, test_ids, train_MS, train_ids, top_k=100):
    
    topk_results = {}

    # Pre-normalize training MS vectors for cosine similarity
    train_MS = np.asarray(train_MS)
    train_norm = np.linalg.norm(train_MS, axis=1, keepdims=True)
    train_MS_normalized = train_MS / (train_norm + 1e-10)

    for i, (test_vec, test_id) in enumerate(zip(test_MS, test_ids)):
        test_vec = np.asarray(test_vec).reshape(1, -1)
        test_vec_norm = np.linalg.norm(test_vec)
        if test_vec_norm == 0:
            similarities = np.zeros(train_MS.shape[0])
        else:
            test_vec_normalized = test_vec / test_vec_norm
            similarities = np.dot(train_MS_normalized, test_vec_normalized.T).ravel()

        # Get indices of top-k similar training samples
        topk_idx = np.argpartition(-similarities, top_k)[:top_k]
        topk_sorted = topk_idx[np.argsort(-similarities[topk_idx])]
        
        topk_results[test_id] = [(train_ids[j], similarities[j]) for j in topk_sorted]

    return topk_results

def string_to_bits(string): 

    bits = np.array([int(c) for c in string])

    return bits

if __name__ == "__main__":

    frags_folder = "/data/rbg/users/klingmin/projects/MS_processing/data/"
    data_folder = "./cache/DreaMS_emb"
    splits_folder = "/data/rbg/users/klingmin/projects/MS_processing/data_splits"
    cache_folder = "./cache/baselines"
    if not os.path.exists(cache_folder): os.makedirs(cache_folder)

    datasets = ["canopus", "massspecgym", "nist2023"]
    splits = ["inchikey_vanilla", "scaffold_vanilla", "random", "LS"]
    setting = "same_CF"

    all_splits = {} 

    for dataset in datasets:

        all_splits[dataset] = {} 

        for split in splits: 

            current_filepath = os.path.join(splits_folder, dataset, "splits", f"{split}.json")
            assert os.path.exists(current_filepath)

            split_ids = load_json(current_filepath)
            train, val, test = split_ids["train"], split_ids["val"], split_ids["test"]
            train = [t.replace(".pkl", "") for t in train]
            test = [t.replace(".pkl", "") for t in test]

            all_splits[dataset][split] = {"train": train,
                                          "test": test}
            
    for dataset in datasets:

        if dataset != "nist2023": continue

        for split in splits: 

            if dataset != "nist2023":
                
                output_path = os.path.join(cache_folder, f"{dataset}_{split}_w_emb.pkl")
                if os.path.exists(output_path): continue 

                train_id = all_splits[dataset][split]["train"]
                test_id = all_splits[dataset][split]["test"]

                train = load_pickle(os.path.join(data_folder, dataset, split, "train.pkl"))
                test = load_pickle(os.path.join(data_folder, dataset, split, "test.pkl"))

                similarity = cosine_similarity(test, train)
                pickle_data((similarity, test_id, train_id), output_path)
            
            else:

                if setting == "same_CF":

                    print(f"Processing {dataset}, {split} split now")

                    train_ids = all_splits[dataset][split]["train"]
                    test_ids = all_splits[dataset][split]["test"]
                    train = load_pickle(os.path.join(data_folder, dataset, split, "train.pkl"))
                    test = load_pickle(os.path.join(data_folder, dataset, split, "test.pkl"))

                    output_folder = os.path.join(cache_folder, f"{dataset}_{split}_w_emb_same_CF_batched")
                    if not os.path.exists(output_folder): os.makedirs(output_folder) 

                    batch_size = 20000

                    test_rec = [load_pickle(os.path.join(frags_folder, dataset, "frags_preds", t + ".pkl")) for t in test_ids]
                    test_CF = set([rec["formula"] for rec in test_rec])

                    for start in range(0, len(train), batch_size):
                        
                        end = min(start + batch_size, len(train))
                        output_path = os.path.join(output_folder, f"{start}_{end}.pkl")
                        if os.path.exists(output_path): continue 
                        
                        current_train_ids = train_ids[start:end]
                        train_rec = [load_pickle(os.path.join(frags_folder, dataset, "frags_preds", t + ".pkl")) for t in current_train_ids]
                        train_formulas = [r["formula"] for r in train_rec]

                        # Get the list of formula to rec
                        formula_idx_mapping = defaultdict(list)
                        for i, f in enumerate(train_formulas): formula_idx_mapping[f].append(i)

                        similarity = cosine_similarity(test, train[start:end])
                        
                        # Now get the top_train_ids and sim lists
                        top_train_ids, sim = [],[]
                        for idx, test_id in enumerate(test_ids):
                            current_sim = similarity[idx, :]
                            train_rows = formula_idx_mapping.get(test_rec[idx]["formula"], None)
                            if train_rows is None: 
                                top_train_ids.append("-")
                                sim.append(0)
                            
                            else: 
                                best_idx = int(np.argmax(current_sim[train_rows]))
                                top_train_ids.append(current_train_ids[best_idx])
                                sim.append(current_sim[best_idx])
                        
                        pickle_data((test_ids, top_train_ids, sim), output_path)

                else: 

                    print(f"Processing {dataset}, {split} split now")

                    train_id = all_splits[dataset][split]["train"]
                    test_id = all_splits[dataset][split]["test"]
                    train = load_pickle(os.path.join(data_folder, dataset, split, "train.pkl"))
                    test = load_pickle(os.path.join(data_folder, dataset, split, "test.pkl"))

                    output_folder = os.path.join(cache_folder, f"{dataset}_{split}_w_emb_batched")
                    if not os.path.exists(output_folder): os.makedirs(output_folder) 

                    batch_size = 20000
                    
                    for start in range(0, len(train), batch_size):
                        
                        end = min(start + batch_size, len(train))
                        output_path = os.path.join(output_folder, f"{start}_{end}.pkl")
                        if os.path.exists(output_path): continue 
                        
                        similarity = cosine_similarity(test, train[start:end])
                        idx = np.argmax(similarity, axis = 1)
                        train_id_batch = train_id[start:end]
                        top_train_id = [train_id_batch[idx[i]] for i in range(len(idx))]
                        sim = [similarity[i][idx[i]] for i in range(len(idx))]

                        pickle_data((test_id, top_train_id, sim), output_path)

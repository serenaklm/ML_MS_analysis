import os 
import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_pickle, load_json, pickle_data

# For processing the MS 
def bin_MS(peaks, bin_resolution = 0.25, max_da = 2000):

    mz = [p["mz"] for p in peaks]
    intensities = [p["intensity"] for p in peaks]
    
    n_bins = int(math.ceil(max_da / bin_resolution))

    mz_binned = [0 for _ in range(n_bins)]
    for m, i in zip(mz, intensities):
        
        m = math.floor(m / bin_resolution)
        if m >= n_bins: continue 
        mz_binned[m] += i

    return mz_binned

# For processing the MS 
def bin_MS_binary(peaks, bin_resolution = 0.25, max_da = 2000):

    mz = [p["mz"] for p in peaks]
    intensities = [p["intensity"] for p in peaks]
    
    n_bins = int(math.ceil(max_da / bin_resolution))

    mz_binned = [0 for _ in range(n_bins)]
    for m, i in zip(mz, intensities):
        
        m = math.floor(m / bin_resolution)
        if m >= n_bins: continue 
        mz_binned[m] = 1

    return mz_binned

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

    data_folder = "/data/rbg/users/klingmin/projects/MS_processing/data/"
    splits_folder = "/data/rbg/users/klingmin/projects/MS_processing/data_splits"
    cache_folder = "./cache/baselines"
    if not os.path.exists(cache_folder): os.makedirs(cache_folder)

    datasets = ["canopus", "massspecgym", "nist2023"]
    splits = ["LS"]
    setting = "same_CF" # "same_CF" or "all"

    all_splits = {} 

    for dataset in datasets:

        all_splits[dataset] = {} 

        for split in splits: 

            current_filepath = os.path.join(splits_folder, dataset, "splits", f"{split}.json")
            assert os.path.exists(current_filepath)

            split_ids = load_json(current_filepath)
            train, test = split_ids["train"], split_ids["test"]
            train = [t.replace(".pkl", "") for t in train]
            test = [t.replace(".pkl", "") for t in test]

            all_splits[dataset][split] = {"train": train,
                                          "test": test}

    dataset_info = {} 

    # canopus = load_pickle(os.path.join(data_folder, "canopus", "canopus_w_mol_info_w_frag_CF_preds.pkl"))
    # canopus = {str(r["id_"]) : r for r in canopus}
    # print("Done loading canopus")

    # massspecgym = load_pickle(os.path.join(data_folder, "massspecgym", "massspecgym_w_mol_info_w_frag_CF_preds.pkl"))
    # massspecgym = {str(r["id_"]) : r for r in massspecgym}
    # print("Done loading MSG")

    # dataset_info["canopus"] = canopus
    # dataset_info["massspecgym"] = massspecgym

    for dataset in datasets:

        if dataset != "nist2023": continue

        for split in splits: 

            if dataset != "nist2023":

                output_path = os.path.join(cache_folder, f"{dataset}_{split}.pkl")
                if os.path.exists(output_path): continue 

                print(f"Processing {dataset}, {split} split now")
                train, test = all_splits[dataset][split]["train"], all_splits[dataset][split]["test"]

                train_MS = [bin_MS(dataset_info[dataset][id_]["peaks"]) for id_ in train] 
                test_MS = [bin_MS(dataset_info[dataset][id_]["peaks"]) for id_ in test] 

                similarity = cosine_similarity(test_MS, train_MS)
                pickle_data((similarity, test, train), output_path)
            
            else:
                
                if setting == "same_CF":
                    
                    print(f"Processing {dataset}, {split} split (Only same CF) now")
                    train_ids, test_ids = all_splits[dataset][split]["train"], all_splits[dataset][split]["test"]

                    output_folder = os.path.join(cache_folder, f"{dataset}_{split}_same_CF_batched")
                    if not os.path.exists(output_folder): os.makedirs(output_folder)

                    batch_size = 20000
                    test_rec = [load_pickle(os.path.join(data_folder, dataset, "frags_preds", t + ".pkl")) for t in test_ids]
                    test_MS = [bin_MS(rec["peaks"]) for rec in test_rec]
                    test_CF = set([rec["formula"] for rec in test_rec])

                    for start in tqdm(range(0, len(train), batch_size)):

                        end = min(start + batch_size, len(train))

                        current_train_ids = train_ids[start:end]

                        output_path = os.path.join(output_folder, f"{start}_{end}.pkl")
                        if os.path.exists(output_path): continue

                        train_rec = [load_pickle(os.path.join(data_folder, dataset, "frags_preds", t + ".pkl")) for t in current_train_ids]
                        kept_idx = [i for i in range(len(current_train_ids)) if train_rec[i]["formula"] in test_CF]

                        # Update everything based on the filtered idx
                        train_rec = [train_rec[i] for i in kept_idx]
                        train_MS = [bin_MS(r["peaks"]) for r in train_rec]
                        current_train_ids = [current_train_ids[i] for i in kept_idx]

                        train_formulas = [r["formula"] for r in train_rec]

                        # Get the list of formula to rec
                        formula_idx_mapping = defaultdict(list)
                        for i, f in enumerate(train_formulas): formula_idx_mapping[f].append(i)

                        # Now get the top k of the same formula 
                        similarity = cosine_similarity(test_MS, train_MS)

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

                if setting == "all":
                        
                    print(f"Processing {dataset}, {split} split now")
                    train_ids, test_ids = all_splits[dataset][split]["train"], all_splits[dataset][split]["test"]

                    output_folder = os.path.join(cache_folder, f"{dataset}_{split}_batched")
                    if not os.path.exists(output_folder): os.makedirs(output_folder)

                    batch_size = 20000
                    test_MS = [bin_MS(load_pickle(os.path.join(data_folder, dataset, "frags_preds", t + ".pkl"))["peaks"]) for t in test_ids]

                    for start in tqdm(range(0, len(train), batch_size)):

                        end = min(start + batch_size, len(train))

                        current_train_ids = train_ids[start:end]

                        output_path = os.path.join(output_folder, f"{start}_{end}.pkl")
                        if os.path.exists(output_path): continue

                        train_MS = [bin_MS(load_pickle(os.path.join(data_folder, dataset, "frags_preds", t + ".pkl"))["peaks"]) for t in current_train_ids]

                        similarity = cosine_similarity(test_MS, train_MS)
                        idx = np.argmax(similarity, axis = 1)
                        top_train_ids = [current_train_ids[idx[i]] for i in range(len(idx))]
                        sim = [similarity[i][idx[i]] for i in range(len(idx))]
                        
                        pickle_data((test_ids, top_train_ids, sim), output_path)

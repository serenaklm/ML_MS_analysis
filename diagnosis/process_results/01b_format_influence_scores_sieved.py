import os
import pickle
import numpy as np
from tqdm import tqdm 
import scipy.stats as stats

from utils import load_pickle, pickle_data
from utils import compute_mol_sim, compute_MS_sim, get_matched_peaks, get_matched_CF

import torch

def get_statistical_diff(helpful, harmful):

    t_statistic, two_sided_p_value = stats.ttest_ind(helpful, harmful)

    # Test if the mean of helpful is less than the mean of harmful
    _, p_val_harmful_more = stats.ttest_ind(helpful, harmful, alternative='less')

    # Test if the mean of helpful is more than the mean of harmful
    _, p_val_helpful_more = stats.ttest_ind(helpful, harmful, alternative='greater')

    return two_sided_p_value, p_val_helpful_more, p_val_harmful_more

def get_stats_cases(IF_cache_folder, f, top_k_test, top_k_train):

    dataset = f.split("/")[-2].replace("_sieved", "")
    
    current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
    if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
    good_info_output_path = os.path.join(current_output_folder, "good_info.pkl")
    bad_info_output_path = os.path.join(current_output_folder, "bad_info.pkl")

    if os.path.exists(good_info_output_path) and os.path.exists(bad_info_output_path): return
    
    IF_scores_path = os.path.join(f, "EK-FAC_scores.pkl")
    IF_scores = load_pickle(IF_scores_path)["all_modules"]
    test_id_list = [i.replace(".ms", "").split("/")[-1] for i in load_pickle(os.path.join(f, "test_ids.pkl"))]
    train_id_list = [i.replace(".ms", "").split("/")[-1] for i in load_pickle(os.path.join(f, "train_ids.pkl"))]

    test_id_list = [k if k.endswith(".pkl") else k + ".pkl" for k in test_id_list]
    train_id_list = [k if k.endswith(".pkl") else k + ".pkl" for k in train_id_list]

    test_results = load_pickle(os.path.join(f, "test_results.pkl"))
    sorted_items = sorted(test_results.items(), key=lambda item: item[1]["loss"])
    ranking = {k: rank for rank, (k, _) in enumerate(sorted_items, start=1)}

    good_test = [k for k,v in ranking.items() if v <= top_k_test]
    bad_test = [k for k,v in ranking.items() if v >= len(sorted_items) - top_k_test]

    good_test = [k if k.endswith(".pkl") else k + ".pkl" for k in good_test]
    bad_test = [k if k.endswith(".pkl") else k + ".pkl" for k in bad_test]

    info_good, info_bad = [], [] 

    for test_idx in range(len(good_test)):

        test_path = os.path.join(data_folder, dataset, "frags_preds", good_test[test_idx])
        test_info = load_pickle(test_path)

        test_idx = test_id_list.index(bad_test[test_idx])
        current_scores = IF_scores[test_idx, :]
        top_k_harmful = torch.topk(-current_scores, k = top_k_train)
        top_k_harmful = top_k_harmful.indices

        top_k_helpful = torch.topk(current_scores, k = top_k_train)
        top_k_helpful = top_k_helpful.indices

        mol_similarity_harmful_good, MS_similarity_harmful_good, CF_overlap_harmful_good = [], [], []
        
        for i, train_idx in enumerate(top_k_harmful.numpy()):

            score = IF_scores[test_idx, train_idx].item()
            if score > 0: continue
            train_path = os.path.join(data_folder, dataset, "frags_preds", train_id_list[train_idx])
            train_info = load_pickle(train_path)

            test_smiles = test_info["smiles"]
            train_smiles = train_info["smiles"] 

            mol_similarity_harmful = compute_mol_sim(test_smiles, train_smiles)
            MS_similarity_harmful = compute_MS_sim(test_info, train_info)
            CF_overlap_harmful = get_matched_CF(test_info, train_info)
            
            mol_similarity_harmful_good.append(mol_similarity_harmful)
            MS_similarity_harmful_good.append(MS_similarity_harmful)
            CF_overlap_harmful_good.append(CF_overlap_harmful)

        mol_similarity_helpful_good, MS_similarity_helpful_good, CF_overlap_helpful_good = [], [], []

        for i, train_idx in enumerate(top_k_helpful.numpy()):

            score = IF_scores[test_idx, train_idx].item()
            if score < 0: continue
            train_path = os.path.join(data_folder, dataset, "frags_preds", train_id_list[train_idx])
            train_info = load_pickle(train_path)

            test_smiles = test_info["smiles"]
            train_smiles = train_info["smiles"]
            
            mol_similarity_helpful = compute_mol_sim(test_smiles, train_smiles)
            MS_similarity_helpful = compute_MS_sim(test_info, train_info)
            CF_overlap_helpful = get_matched_CF(test_info, train_info)
            
            mol_similarity_helpful_good.append(mol_similarity_helpful)
            MS_similarity_helpful_good.append(MS_similarity_helpful)
            CF_overlap_helpful_good.append(CF_overlap_helpful)

        # Let us get some statistics now
        mean_mol_similarity_helpful_good = np.mean(mol_similarity_helpful_good)
        mean_mol_similarity_harmful_good = np.mean(mol_similarity_harmful_good)
        mol_similarity_diff_p_value_good_list = get_statistical_diff(mol_similarity_helpful_good, mol_similarity_harmful_good)
        mol_similarity_two_sided_p_value_good, \
        mol_similarity_one_sided_p_value_helpful_more,\
        mol_similarity_one_sided_p_value_harmful_more = mol_similarity_diff_p_value_good_list
        
        mean_MS_similarity_helpful_good = np.mean(MS_similarity_helpful_good)
        mean_MS_similarity_harmful_good = np.mean(MS_similarity_harmful_good)
        MS_similarity_diff_p_value_good_list = get_statistical_diff(MS_similarity_helpful_good, MS_similarity_harmful_good)
        MS_similarity_two_sided_p_value_good, \
        MS_similarity_one_sided_p_value_helpful_more,\
        MS_similarity_one_sided_p_value_harmful_more = MS_similarity_diff_p_value_good_list

        mean_CF_overlap_helpful_good = np.mean(CF_overlap_helpful_good)
        mean_CF_overlap_harmful_good = np.mean(CF_overlap_harmful_good)
        CF_overlap_diff_p_value_good_list = get_statistical_diff(CF_overlap_helpful_good, CF_overlap_harmful_good)
        CF_overlap_two_sided_p_value_good, \
        CF_overlap_one_sided_p_value_helpful_more,\
        CF_overlap_one_sided_p_value_harmful_more = CF_overlap_diff_p_value_good_list

        # We do some labelling now
        if mol_similarity_one_sided_p_value_harmful_more < p_value_threshold: mol_similarity_label = "harmful_more"
        elif mol_similarity_one_sided_p_value_helpful_more < p_value_threshold: mol_similarity_label = "helpful_more"
        else: mol_similarity_label = "no_diff"

        if MS_similarity_one_sided_p_value_harmful_more < p_value_threshold: MS_similarity_label = "harmful_more"
        elif MS_similarity_one_sided_p_value_helpful_more < p_value_threshold: MS_similarity_label = "helpful_more"
        else: MS_similarity_label = "no_diff"

        final_label = f"mol_{mol_similarity_label}_MS_{MS_similarity_label}"

        info_good.append([mean_mol_similarity_helpful_good, mean_mol_similarity_harmful_good, mol_similarity_label, \
                        mol_similarity_two_sided_p_value_good, mol_similarity_one_sided_p_value_helpful_more, mol_similarity_one_sided_p_value_harmful_more,\
                        mean_MS_similarity_helpful_good, mean_MS_similarity_harmful_good, MS_similarity_label, \
                        MS_similarity_two_sided_p_value_good, MS_similarity_one_sided_p_value_helpful_more, MS_similarity_one_sided_p_value_harmful_more,\
                        mean_CF_overlap_helpful_good, mean_CF_overlap_harmful_good, \
                        CF_overlap_two_sided_p_value_good, CF_overlap_one_sided_p_value_helpful_more, CF_overlap_one_sided_p_value_harmful_more, final_label])

    # Save info on the good predictions 
    pickle_data(info_good, good_info_output_path)

    for test_idx in range(len(bad_test)):

        test_path = os.path.join(data_folder, dataset, "frags_preds", bad_test[test_idx])
        test_info = load_pickle(test_path)
        test_idx = test_id_list.index(bad_test[test_idx])
        current_scores = IF_scores[test_idx, :]

        top_k_harmful = torch.topk(-current_scores, k = top_k_train)
        top_k_harmful = top_k_harmful.indices

        top_k_helpful = torch.topk(current_scores, k = top_k_train)
        top_k_helpful = top_k_helpful.indices

        mol_similarity_harmful_bad, MS_similarity_harmful_bad, CF_overlap_harmful_bad = [], [], []

        for i, train_idx in enumerate(top_k_harmful.numpy()):

            score = IF_scores[test_idx, train_idx].item()
            if score > 0: continue
            train_path = os.path.join(data_folder, dataset, "frags_preds", train_id_list[train_idx])
            train_info = load_pickle(train_path)

            test_smiles = test_info["smiles"]
            train_smiles = train_info["smiles"] 

            mol_similarity_harmful = compute_mol_sim(test_smiles, train_smiles)
            MS_similarity_harmful = compute_MS_sim(test_info, train_info)
            CF_overlap_harmful = get_matched_CF(test_info, train_info)

            mol_similarity_harmful_bad.append(mol_similarity_harmful)
            MS_similarity_harmful_bad.append(MS_similarity_harmful)
            CF_overlap_harmful_bad.append(CF_overlap_harmful)

        mol_similarity_helpful_bad, MS_similarity_helpful_bad, CF_overlap_helpful_bad = [], [], []

        for i, train_idx in enumerate(top_k_helpful.numpy()):

            score = IF_scores[test_idx, train_idx].item()
            if score < 0: continue
            train_path = os.path.join(data_folder, dataset, "frags_preds", train_id_list[train_idx])
            train_info = load_pickle(train_path)

            test_smiles = test_info["smiles"]
            train_smiles = train_info["smiles"] 

            mol_similarity_helpful = compute_mol_sim(test_smiles, train_smiles)
            MS_similarity_helpful = compute_MS_sim(test_info, train_info)
            CF_overlap_helpful = get_matched_CF(test_info, train_info)
            
            mol_similarity_helpful_bad.append(mol_similarity_helpful)
            MS_similarity_helpful_bad.append(MS_similarity_helpful)
            CF_overlap_helpful_bad.append(CF_overlap_helpful)

        # Let us get some statistics now
        mean_mol_similarity_helpful_bad = np.mean(mol_similarity_helpful_bad)
        mean_mol_similarity_harmful_bad = np.mean(mol_similarity_harmful_bad)
        mol_similarity_diff_p_value_bad_list = get_statistical_diff(mol_similarity_helpful_bad, mol_similarity_harmful_bad)
        mol_similarity_two_sided_p_value_bad, \
        mol_similarity_one_sided_p_value_helpful_more,\
        mol_similarity_one_sided_p_value_harmful_more = mol_similarity_diff_p_value_bad_list
        
        mean_MS_similarity_helpful_bad = np.mean(MS_similarity_helpful_bad)
        mean_MS_similarity_harmful_bad = np.mean(MS_similarity_harmful_bad)
        MS_similarity_diff_p_value_bad_list = get_statistical_diff(MS_similarity_helpful_bad, MS_similarity_harmful_bad)
        MS_similarity_two_sided_p_value_bad, \
        MS_similarity_one_sided_p_value_helpful_more,\
        MS_similarity_one_sided_p_value_harmful_more = MS_similarity_diff_p_value_bad_list

        mean_CF_overlap_helpful_bad = np.mean(CF_overlap_helpful_bad)
        mean_CF_overlap_harmful_bad = np.mean(CF_overlap_harmful_bad)
        CF_overlap_diff_p_value_bad_list = get_statistical_diff(CF_overlap_helpful_bad, CF_overlap_harmful_bad)
        CF_overlap_two_sided_p_value_bad, \
        CF_overlap_one_sided_p_value_helpful_more,\
        CF_overlap_one_sided_p_value_harmful_more = CF_overlap_diff_p_value_bad_list

        # We do some labelling now
        if mol_similarity_one_sided_p_value_harmful_more < p_value_threshold: mol_similarity_label = "harmful_more"
        elif mol_similarity_one_sided_p_value_helpful_more < p_value_threshold: mol_similarity_label = "helpful_more"
        else: mol_similarity_label = "no_diff"

        if MS_similarity_one_sided_p_value_harmful_more < p_value_threshold: MS_similarity_label = "harmful_more"
        elif MS_similarity_one_sided_p_value_helpful_more < p_value_threshold: MS_similarity_label = "helpful_more"
        else: MS_similarity_label = "no_diff"

        final_label = f"mol_{mol_similarity_label}_MS_{MS_similarity_label}"

        info_bad.append([mean_mol_similarity_helpful_bad, mean_mol_similarity_harmful_bad, mol_similarity_label, \
                         mol_similarity_two_sided_p_value_bad, mol_similarity_one_sided_p_value_helpful_more, mol_similarity_one_sided_p_value_harmful_more,\
                         mean_MS_similarity_helpful_bad, mean_MS_similarity_harmful_bad, MS_similarity_label, \
                         MS_similarity_two_sided_p_value_bad, MS_similarity_one_sided_p_value_helpful_more, MS_similarity_one_sided_p_value_harmful_more,\
                         mean_CF_overlap_helpful_bad, mean_CF_overlap_harmful_bad, \
                         CF_overlap_two_sided_p_value_bad, CF_overlap_one_sided_p_value_helpful_more, CF_overlap_one_sided_p_value_harmful_more, final_label])

    # Save info on the bad predictions 
    pickle_data(info_bad, bad_info_output_path)

if __name__ == "__main__":

    top_k_test, top_k_train = 500, 500
    p_value_threshold = 0.05
    cache_folder = "../cache"
    IF_cache_folder = os.path.join(cache_folder, "IF_results")
    if not os.path.exists(IF_cache_folder): os.makedirs(IF_cache_folder)
    
    mist_folder = "../../FP_prediction/mist/best_models"
    baseline_models_folder = "../../FP_prediction/baseline_models/best_models"
    data_folder = "/data/rbg/users/klingmin/projects/MS_processing/data"

    results_folder = []

    for folder in [baseline_models_folder, mist_folder]:
        for dataset in os.listdir(folder):
            for checkpoint in os.listdir(os.path.join(folder, dataset)):
                if "sieved" not in checkpoint: continue
                IF_score_path = os.path.join(folder, dataset, checkpoint, "EK-FAC_scores.pkl")
                if not os.path.exists(IF_score_path): continue
                results_folder.append(os.path.join(folder, dataset, checkpoint))
    
    # Iterate through the results file 
    for f in results_folder:
        print(f"Processing {f}")
        get_stats_cases(IF_cache_folder, f, top_k_test, top_k_train)
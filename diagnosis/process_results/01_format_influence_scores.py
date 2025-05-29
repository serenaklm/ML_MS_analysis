import os
import pickle
import numpy as np
from tqdm import tqdm 

from utils import load_pickle, pickle_data
from utils import compute_mol_sim, compute_MS_sim, get_matched_peaks, get_matched_CF

import torch

def bin_energy(e):
    
    if e is None: return "-"
    elif e == "-": return "-"
    elif e < 20: return "0-20"
    elif e < 40: return "20-40"
    elif e < 60: return "40-60"
    elif e < 80: return "60-80"
    elif e < 100: return "80-100"
    elif e < 120: return "100-120"
    elif e < 150: return "120-150"
    else: return "150-"

def get_index_of_identical_mol(current_labels, test_id, train_id_list):

    test_id = test_id.split("/")[-1].replace(".pkl", "")
    train_id_list = [t.split("/")[-1].replace(".pkl", "") for t in train_id_list]
    test_mol = current_labels[test_id]["inchikey_original"][:14]
    train_idx = [i for i, train_id in enumerate(train_id_list) if current_labels[train_id]["inchikey_original"][:14] == test_mol]

    return train_idx

def get_index_of_identical_mol_diff_expt_cond(current_labels, test_id, train_id_list):

    test_id = test_id.split("/")[-1].replace(".pkl", "")
    test_rec = current_labels[test_id]
    train_id_list = [t.split("/")[-1].replace(".pkl", "") for t in train_id_list]
    train_id_list_sieved = [] 

    for i, train_id in enumerate(train_id_list):

        train_rec = current_labels[train_id]
        if train_rec["inchikey_original"][:14] != test_rec["inchikey_original"][:14]: continue
        if train_rec["instrument_type"] != test_rec["instrument_type"] \
        or train_rec["precursor_type"] != test_rec["precursor_type"] \
        or bin_energy(train_rec["collision_energy"]) != bin_energy(test_rec["collision_energy"]):
            train_id_list_sieved.append(i)

    return train_id_list_sieved 

def get_mol_sim(current_labels, test_id, train_id_list):

    test_id = test_id.split("/")[-1].replace(".pkl", "")
    train_id_list = [t.split("/")[-1].replace(".pkl", "") for t in train_id_list]

    test_smiles = current_labels[test_id]["smiles"]
    all_sim = [] 

    for train_id in train_id_list:

        train_smiles = current_labels[train_id]["smiles"]
        sim = compute_mol_sim(test_smiles, train_smiles)
        all_sim.append(sim)
    
    return all_sim

def get_MS_sim(frags_folder, test_id, train_id_list):

    test_id = test_id.split("/")[-1].replace(".pkl", "")
    train_id_list = [t.split("/")[-1].replace(".pkl", "") for t in train_id_list]

    test_rec = load_pickle(os.path.join(frags_folder, f"{test_id}.pkl"))

    all_sim = [] 

    for train_id in train_id_list:

        train_rec = load_pickle(os.path.join(frags_folder, f"{train_id}.pkl"))
        sim = compute_MS_sim(test_rec, train_rec)
        all_sim.append(sim)
    
    return all_sim

def get_peaks_formula_con(frags_folder, test_id, train_id_list):

    test_id = test_id.split("/")[-1].replace(".pkl", "")
    train_id_list = [t.split("/")[-1].replace(".pkl", "") for t in train_id_list]

    test_rec = load_pickle(os.path.join(frags_folder, f"{test_id}.pkl"))

    all_cons = [] 

    for train_id in train_id_list:

        train_rec = load_pickle(os.path.join(frags_folder, f"{train_id}.pkl"))
        matches = get_matched_peaks(test_rec, train_rec)
        matches = [m for m in matches if m[-1] != "" and m[-2] != ""]

        if len(matches) == 0: 
            all_cons.append(0)
        else:

            frag_matches = [m for m in matches if m[-1] == m[-2]]
            cons = len(frag_matches) / len(matches)
            all_cons.append(cons)
    
    return all_cons

def get_CF_overlap(frags_folder, test_id, train_id_list):

    test_id = test_id.split("/")[-1].replace(".pkl", "")
    train_id_list = [t.split("/")[-1].replace(".pkl", "") for t in train_id_list]

    test_rec = load_pickle(os.path.join(frags_folder, f"{test_id}.pkl"))

    all_overlap = [] 

    for train_id in train_id_list:

        train_rec = load_pickle(os.path.join(frags_folder, f"{train_id}.pkl"))
        percent_overlap = get_matched_CF(test_rec, train_rec)
        all_overlap.append(percent_overlap)

    return all_overlap

def get_identical_mol_percent_harmful_helpful(data_folder, results_folder):
        
    results_folder_random = [f for f in results_folder if "random" in f]

    for f in results_folder_random:
        
        current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
        if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
        output_path = os.path.join(current_output_folder, "identical_mol_helpful_harmful_counts.pkl")

        if not os.path.exists(output_path):
            
            print(f"Processing {f}")
            
            dataset = f.split("/")[-2].replace("_sieved", "")

            current_path = os.path.join(f, "EK-FAC_scores.pkl")
            if not os.path.exists(current_path): continue 

            IF_scores = load_pickle(current_path)["all_modules"]
            train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
            test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]
            current_labels = load_pickle(os.path.join(data_folder, dataset, f"{dataset}_w_mol_info.pkl"))
            current_labels = {str(r["id_"]): r for r in current_labels}

            helpful_harmful_percent = {} 

            for test_idx in tqdm(range(len(test_id_list))):

                test_id = test_id_list[test_idx]
                current_scores = IF_scores[test_idx, :]

                identical_mol_train = get_index_of_identical_mol(current_labels, test_id, train_id_list)
                if len(identical_mol_train) == 0: continue
                identical_mol_helpful = [i for i in identical_mol_train if current_scores[i] > 0]
                identical_mol_harmful = [i for i in identical_mol_train if current_scores[i] < 0]

                percent_mol_helpful = len(identical_mol_helpful) / len(identical_mol_train)
                percent_mol_harmful = len(identical_mol_harmful) / len(identical_mol_train)

                helpful_harmful_percent[test_id] = {"helpful": percent_mol_helpful,
                                                    "harmful": percent_mol_harmful}

            pickle_data(helpful_harmful_percent, output_path)

def get_identical_mol_diff_expt_condition_percent_harmful_helpful(data_folder, results_folder):

    results_folder_random = [f for f in results_folder if "random" in f]

    for f in results_folder_random:
        
        current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
        if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
        output_path = os.path.join(current_output_folder, "identical_mol_diff_expt_condition_helpful_harmful_counts.pkl")

        if not os.path.exists(output_path):
            
            print(f"Processing {f}")
            
            dataset = f.split("/")[-2].replace("_sieved", "")

            current_path = os.path.join(f, "EK-FAC_scores.pkl")
            if not os.path.exists(current_path): continue 

            IF_scores = load_pickle(current_path)["all_modules"]
            train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
            test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]
            current_labels = load_pickle(os.path.join(data_folder, dataset, f"{dataset}_w_mol_info.pkl"))
            current_labels = {str(r["id_"]): r for r in current_labels}

            helpful_harmful_percent = {} 

            for test_idx in tqdm(range(len(test_id_list))):

                test_id = test_id_list[test_idx]
                current_scores = IF_scores[test_idx, :]

                identical_mol_train_diff_expt_cond = get_index_of_identical_mol_diff_expt_cond(current_labels, test_id, train_id_list)
                if len(identical_mol_train_diff_expt_cond) == 0: continue 

                identical_mol_helpful = [i for i in identical_mol_train_diff_expt_cond if current_scores[i] > 0]
                identical_mol_harmful = [i for i in identical_mol_train_diff_expt_cond if current_scores[i] < 0]

                percent_mol_helpful = len(identical_mol_helpful) / len(identical_mol_train_diff_expt_cond)
                percent_mol_harmful = len(identical_mol_harmful) / len(identical_mol_train_diff_expt_cond)

                helpful_harmful_percent[test_id] = {"helpful": percent_mol_helpful,
                                                    "harmful": percent_mol_harmful}
                
            # Save the results
            pickle_data(helpful_harmful_percent, output_path)

def get_mol_sim_harmful_helpful(data_folder, results_folder, top_k):

    for f in results_folder:

        current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
        if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
        output_path = os.path.join(current_output_folder, "mol_sim_helpful_harmful.pkl")
        
        dataset = f.split("/")[-2].replace("_sieved", "")
        current_data_folder = os.path.join(data_folder, dataset)

        if not os.path.exists(output_path):
            
            print(f"Processing {f}")
            
            current_path = os.path.join(f, "EK-FAC_scores.pkl")
            if not os.path.exists(current_path): continue 

            IF_scores = load_pickle(current_path)["all_modules"]
            
            train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
            test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]

            current_labels = load_pickle(os.path.join(current_data_folder, f"{dataset}_w_mol_info.pkl"))
            current_labels = {str(r["id_"]): r for r in current_labels}

            helpful_harmful_mol_sim = {} 

            for test_idx in tqdm(range(len(test_id_list))):

                test_id = test_id_list[test_idx]
                current_scores = IF_scores[test_idx, :]
                _, indices = torch.sort(current_scores)
                top_k_harmful, top_k_helpful = indices[:top_k].numpy().tolist(), indices[-top_k:].numpy().tolist()

                harmful_train = [train_id_list[i] for i in top_k_harmful]
                helpful_train = [train_id_list[i] for i in top_k_helpful]
                
                mol_sim_harmful = get_mol_sim(current_labels, test_id, harmful_train)
                mol_sim_helpful = get_mol_sim(current_labels, test_id, helpful_train)

                helpful_harmful_mol_sim[test_id] = {"helpful": {helpful_train[i]: mol_sim_helpful[i] for i in range(len(helpful_train))},
                                                    "harmful": {harmful_train[i]: mol_sim_harmful[i] for i in range(len(harmful_train))}}
                
            # Save the results
            pickle_data(helpful_harmful_mol_sim, output_path)

def get_MS_sim_harmful_helpful(data_folder, results_folder, top_k):

    for f in results_folder:

        current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
        if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
        output_path = os.path.join(current_output_folder, "MS_sim_helpful_harmful.pkl")
        
        dataset = f.split("/")[-2].replace("_sieved", "")
        current_data_folder = os.path.join(data_folder, dataset, "frags_preds")

        if not os.path.exists(output_path):
            
            print(f"Processing {f}")
            
            current_path = os.path.join(f, "EK-FAC_scores.pkl")
            if not os.path.exists(current_path): continue 

            IF_scores = load_pickle(current_path)["all_modules"]
            
            train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
            test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]

            helpful_harmful_MS_sim = {} 

            for test_idx in tqdm(range(len(test_id_list))):

                test_id = test_id_list[test_idx]
                current_scores = IF_scores[test_idx, :]
                _, indices = torch.sort(current_scores)
                top_k_harmful, top_k_helpful = indices[:top_k].numpy().tolist(), indices[-top_k:].numpy().tolist()

                harmful_train = [train_id_list[i] for i in top_k_harmful]
                helpful_train = [train_id_list[i] for i in top_k_helpful]
                
                MS_sim_harmful = get_MS_sim(current_data_folder, test_id, harmful_train)
                MS_sim_helpful = get_MS_sim(current_data_folder, test_id, helpful_train)

                helpful_harmful_MS_sim[test_id] = {"helpful": {helpful_train[i]: MS_sim_helpful[i] for i in range(len(helpful_train))},
                                                   "harmful": {harmful_train[i]: MS_sim_harmful[i] for i in range(len(harmful_train))}}
                
            # Save the results
            pickle_data(helpful_harmful_MS_sim, output_path)

def get_CF_overlap_harmful_helpful(data_folder, results_folder, top_k):

    for f in results_folder:

        current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
        if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
        output_path = os.path.join(current_output_folder, "CF_overlap_helpful_harmful.pkl")
        
        dataset = f.split("/")[-2].replace("_sieved", "")
        current_data_folder = os.path.join(data_folder, dataset, "frags_preds")

        if not os.path.exists(output_path):
            
            print(f"Processing {f}")
            
            current_path = os.path.join(f, "EK-FAC_scores.pkl")
            if not os.path.exists(current_path): continue 

            IF_scores = load_pickle(current_path)["all_modules"]
            
            train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
            test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]

            helpful_harmful_CF_overlap = {} 

            for test_idx in tqdm(range(len(test_id_list))):

                test_id = test_id_list[test_idx]
                current_scores = IF_scores[test_idx, :]
                _, indices = torch.sort(current_scores)
                top_k_harmful, top_k_helpful = indices[:top_k].numpy().tolist(), indices[-top_k:].numpy().tolist()

                harmful_train = [train_id_list[i] for i in top_k_harmful]
                helpful_train = [train_id_list[i] for i in top_k_helpful]
                
                CF_overlap_harmful = get_CF_overlap(current_data_folder, test_id, harmful_train)
                CF_overlap_helpful = get_CF_overlap(current_data_folder, test_id, helpful_train)

                helpful_harmful_CF_overlap[test_id] = {"helpful": {helpful_train[i]: CF_overlap_helpful[i] for i in range(len(helpful_train))},
                                                       "harmful": {harmful_train[i]: CF_overlap_harmful[i] for i in range(len(harmful_train))}}
                
            # Save the results
            pickle_data(helpful_harmful_CF_overlap, output_path)

def get_agreement_mol_MS_sim_harmful_helpful(results_folder):

    for f in results_folder:

        current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
        if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
        output_path = os.path.join(current_output_folder, "mol_MS_sim_con_helpful_harmful.pkl")
        
        if not os.path.exists(output_path):
            
            print(f"Processing {f}")
            MSsim_path = os.path.join(current_output_folder, "MS_sim_helpful_harmful.pkl")
            molsim_path = os.path.join(current_output_folder, "mol_sim_helpful_harmful.pkl")

            if not os.path.exists(MSsim_path) or not os.path.exists(molsim_path): continue 

            MSsim = load_pickle(MSsim_path)
            molsim = load_pickle(molsim_path)
            
            helpful_harmful_mol_MS_con = {} 

            for test_id, rec_list in tqdm(MSsim.items()):

                helpful_MSsim = rec_list["helpful"]
                harmful_MSsim = rec_list["harmful"]

                helpful_molsim = molsim[test_id]["helpful"]
                harmful_molsim = molsim[test_id]["harmful"]

                helpful_scores = {train_id: helpful_MSsim[train_id] * helpful_molsim[train_id] for train_id in helpful_MSsim.keys()}
                harmful_scores = {train_id: harmful_MSsim[train_id] * harmful_molsim[train_id] for train_id in harmful_MSsim.keys()}

                helpful_harmful_mol_MS_con[test_id] = {"helpful": helpful_scores,
                                                       "harmful": harmful_scores}

            # Save the results
            pickle_data(helpful_harmful_mol_MS_con, output_path)

def get_agreement_MS_peaks_CF_harmful_helpful(data_folder, results_folder, top_k):

    for f in results_folder:
        
        current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
        if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
        output_path = os.path.join(current_output_folder, "peaks_CF_con_helpful_harmful.pkl")
        
        dataset = f.split("/")[-2].replace("_sieved", "")
        current_data_folder = os.path.join(data_folder, dataset, "frags_preds")

        print(output_path, os.path.exists(output_path))
        if not os.path.exists(output_path):
            
            print(f"Processing {f}")
            
            current_path = os.path.join(f, "EK-FAC_scores.pkl")
            if not os.path.exists(current_path): continue 

            IF_scores = load_pickle(current_path)["all_modules"]
            
            train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
            test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]

            helpful_harmful_peaks_CF_cons = {} 

            for test_idx in tqdm(range(len(test_id_list))):

                test_id = test_id_list[test_idx]
                current_scores = IF_scores[test_idx, :]
                _, indices = torch.sort(current_scores)
                top_k_harmful, top_k_helpful = indices[:top_k].numpy().tolist(), indices[-top_k:].numpy().tolist()
                
                harmful_train = [train_id_list[i] for i in top_k_harmful]
                helpful_train = [train_id_list[i] for i in top_k_helpful]
                
                peaks_CF_cons_helpful = get_peaks_formula_con(current_data_folder, test_id, helpful_train)
                peaks_CF_cons_harmful = get_peaks_formula_con(current_data_folder, test_id, harmful_train)
                
                helpful_harmful_peaks_CF_cons[test_id] = {"helpful": {helpful_train[i]: peaks_CF_cons_helpful[i] for i in range(len(helpful_train))},
                                                          "harmful": {harmful_train[i]: peaks_CF_cons_harmful[i] for i in range(len(harmful_train))}}
                

            # Save the results
            pickle_data(helpful_harmful_peaks_CF_cons, output_path)

def get_agreement_CF_molsim(results_folder):
    
    for f in results_folder:
        
        current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
        if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
        output_path = os.path.join(current_output_folder, "CF_molsim_con_helpful_harmful.pkl")

        if not os.path.exists(output_path):
            
            print(f"Processing {f}")
            molsim_path = os.path.join(current_output_folder, "mol_sim_helpful_harmful.pkl")

            if not os.path.exists(molsim_path): continue 

            molsim = load_pickle(molsim_path)
            
            helpful_harmful_CF_molsim_cons = {} 

            for test_id, rec_list in tqdm(molsim.items()):

                helpful_mol_sim = molsim[test_id]["helpful"]
                harmful_mol_sim = molsim[test_id]["harmful"]

                # Compute the helpful / harmful CF overlap
                CF_overlap_path = os.path.join(current_output_folder, "CF_overlap_helpful_harmful.pkl")
                mol_sim_path = os.path.join(current_output_folder, "mol_sim_helpful_harmful.pkl")

                CF_overlap = load_pickle(CF_overlap_path)
                mol_sim = load_pickle(mol_sim_path)

                for test_id, rec_list in tqdm(CF_overlap.items()):

                    helpful_CF_overlap = rec_list["helpful"]
                    harmful_CF_overlap = rec_list["harmful"]

                    helpful_mol_sim = mol_sim[test_id]["helpful"]
                    harmful_mol_sim = mol_sim[test_id]["harmful"]

                    helpful_scores = {train_id: helpful_CF_overlap[train_id] * helpful_mol_sim[train_id] for train_id in helpful_CF_overlap.keys()}
                    harmful_scores = {train_id: harmful_CF_overlap[train_id] * harmful_mol_sim[train_id] for train_id in harmful_CF_overlap.keys()}
                    
                    helpful_harmful_CF_molsim_cons[test_id] = {"helpful": helpful_scores,
                                                               "harmful": harmful_scores}
                
            # Save the results
            pickle_data(helpful_harmful_CF_molsim_cons, output_path)

if __name__ == "__main__":

    top_k = 100
    cache_folder = "../cache"
    IF_cache_folder = os.path.join(cache_folder, "IF_results")
    if not os.path.exists(IF_cache_folder): os.makedirs(IF_cache_folder)
    
    mist_folder = "../../FP_prediction/mist/best_models"
    baseline_models_folder = "../../FP_prediction/baseline_models/best_models"
    data_folder = "/data/rbg/users/klingmin/projects/MS_processing/data"

    results_folder = []

    for folder in [mist_folder, baseline_models_folder]:
        for dataset in os.listdir(folder):
            if "sampled" in dataset: continue 
            for checkpoint in os.listdir(os.path.join(folder, dataset)):
                results_folder.append(os.path.join(folder, dataset, checkpoint))
                
    # 1. Look at the percentage of the time where identical molecules are helpful / harmful
    get_identical_mol_percent_harmful_helpful(data_folder, results_folder)

    # 2. Look at the percentage of the time where identical molecules from different experimental conditions are helpful / harmful 
    get_identical_mol_diff_expt_condition_percent_harmful_helpful(data_folder, results_folder)

    # 3. Look at the mol sim of the top k most harmful / helpful molecules 
    get_mol_sim_harmful_helpful(data_folder, results_folder, top_k)

    # 4. Look at the MS sim of the top k most harmful / helpful molecules 
    get_MS_sim_harmful_helpful(data_folder, results_folder, top_k)

    # 5. Look at the CF overlap of the top k most harmful / helpful molecules 
    get_CF_overlap_harmful_helpful(data_folder, results_folder, top_k)

    # # 5. Look at the consistency between the mol sim and MS sim 
    # get_agreement_mol_MS_sim_harmful_helpful(results_folder)

    # # 6. Look at the consistency between the matched peaks and the formula of the matched peaks 
    # get_agreement_MS_peaks_CF_harmful_helpful(data_folder, results_folder, top_k)

    # # 7. Look at the consistency between CF with molecule similarity 
    # get_agreement_CF_molsim(results_folder)

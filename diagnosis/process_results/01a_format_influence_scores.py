import os
import pickle
import numpy as np
from tqdm import tqdm 
from pprint import pprint 

from utils import load_pickle, pickle_data
from utils import compute_MS_sim

import torch

# def get_mol_sim(current_labels, test_id, train_id_list):

#     test_id = test_id.split("/")[-1].replace(".pkl", "")
#     train_id_list = [t.split("/")[-1].replace(".pkl", "") for t in train_id_list]

#     test_smiles = current_labels[test_id]["smiles"]
#     all_sim = [] 

#     for train_id in train_id_list:

#         train_smiles = current_labels[train_id]["smiles"]
#         sim = compute_mol_sim(test_smiles, train_smiles)
#         all_sim.append(sim)
    
#     return all_sim

# def get_MS_sim(frags_folder, test_id, train_id_list):

#     test_id = test_id.split("/")[-1].replace(".pkl", "")
#     train_id_list = [str(t).split("/")[-1].replace(".pkl", "") for t in train_id_list]

#     test_rec = load_pickle(os.path.join(frags_folder, f"{test_id}.pkl"))

#     all_sim = [] 

#     for train_id in train_id_list:

#         train_rec = load_pickle(os.path.join(frags_folder, f"{train_id}.pkl"))
#         sim = compute_MS_sim(test_rec, train_rec)
#         all_sim.append(sim)
    
#     return all_sim

# def get_CF_overlap(frags_folder, test_id, train_id_list):

#     test_id = test_id.split("/")[-1].replace(".pkl", "")
#     train_id_list = [t.split("/")[-1].replace(".pkl", "") for t in train_id_list]

#     test_rec = load_pickle(os.path.join(frags_folder, f"{test_id}.pkl"))

#     all_overlap = [] 

#     for train_id in train_id_list:

#         train_rec = load_pickle(os.path.join(frags_folder, f"{train_id}.pkl"))
#         percent_overlap = get_matched_CF(test_rec, train_rec)
#         all_overlap.append(percent_overlap)

#     return all_overlap

# def get_identical_mol_percent_harmful_helpful(data_folder, results_folder, threshold = 0.0):
        
#     results_folder_random = [f for f in results_folder if "random" in f]

#     for f in results_folder_random:
        
#         current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
#         if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
#         output_path = os.path.join(current_output_folder, "identical_mol_helpful_harmful_counts.pkl")

#         if not os.path.exists(output_path):
            
#             print(f"Processing {f}")
            
#             dataset = f.split("/")[-2].replace("_sieved", "")

#             current_path = os.path.join(f, "EK-FAC_scores.pkl")
#             if not os.path.exists(current_path): continue 

#             IF_scores = load_pickle(current_path)["all_modules"]
#             train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
#             test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]
#             current_labels = load_pickle(os.path.join(data_folder, dataset, f"{dataset}_w_mol_info.pkl"))
#             current_labels = {str(r["id_"]): r for r in current_labels}

#             helpful_harmful_percent = {} 

#             for test_idx in tqdm(range(len(test_id_list))):

#                 test_id = test_id_list[test_idx]
#                 current_scores = IF_scores[test_idx, :]

#                 identical_mol_train = get_index_of_identical_mol(current_labels, test_id, train_id_list)
#                 if len(identical_mol_train) == 0: continue
#                 identical_mol_helpful = [i for i in identical_mol_train if current_scores[i] > 0 and abs(current_scores[i]) > threshold]
#                 identical_mol_harmful = [i for i in identical_mol_train if current_scores[i] < 0 and abs(current_scores[i]) > threshold]

#                 num_mol_helpful = len(identical_mol_helpful)
#                 num_mol_harmful = len(identical_mol_harmful)

#                 helpful_harmful_percent[test_id] = {"helpful": num_mol_helpful,
#                                                     "harmful": num_mol_harmful,
#                                                     "total" : len(identical_mol_train)}

#             pickle_data(helpful_harmful_percent, output_path)

# def get_identical_mol_diff_expt_condition_percent_harmful_helpful(data_folder, results_folder, threshold = 0.0):

#     results_folder_random = [f for f in results_folder if "random" in f]

#     for f in results_folder_random:
        
#         current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
#         if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
#         output_path = os.path.join(current_output_folder, "identical_mol_diff_expt_condition_helpful_harmful_counts.pkl")

#         if not os.path.exists(output_path):
            
#             print(f"Processing {f}")
            
#             dataset = f.split("/")[-2].replace("_sieved", "")

#             current_path = os.path.join(f, "EK-FAC_scores.pkl")
#             if not os.path.exists(current_path): continue 

#             IF_scores = load_pickle(current_path)["all_modules"]
#             train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
#             test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]
#             current_labels = load_pickle(os.path.join(data_folder, dataset, f"{dataset}_w_mol_info.pkl"))
#             current_labels = {str(r["id_"]): r for r in current_labels}

#             helpful_harmful_percent = {} 

#             for test_idx in tqdm(range(len(test_id_list))):

#                 test_id = test_id_list[test_idx]
#                 current_scores = IF_scores[test_idx, :]

#                 identical_mol_train_diff_expt_cond = get_index_of_identical_mol_diff_expt_cond(current_labels, test_id, train_id_list)

#                 if len(identical_mol_train_diff_expt_cond) == 0: continue 

#                 identical_mol_helpful = [i for i in identical_mol_train_diff_expt_cond if current_scores[i] > 0 and abs(current_scores[i]) > threshold]
#                 identical_mol_harmful = [i for i in identical_mol_train_diff_expt_cond if current_scores[i] < 0 and abs(current_scores[i]) > threshold]
                
#                 num_mol_helpful = len(identical_mol_helpful)
#                 num_mol_harmful = len(identical_mol_harmful)

#                 helpful_harmful_percent[test_id] = {"helpful": num_mol_helpful,
#                                                     "harmful": num_mol_harmful,
#                                                     "total": len(identical_mol_train_diff_expt_cond)}
                
#             # Save the results
#             pickle_data(helpful_harmful_percent, output_path)

# def get_identical_mol_identical_expt_condition_percent_harmful_helpful(data_folder, results_folder, threshold = 0.0):

#     results_folder_random = [f for f in results_folder if "random" in f]

#     for f in results_folder_random:
        
#         current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
#         if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
#         output_path = os.path.join(current_output_folder, "identical_mol_identical_expt_condition_helpful_harmful_counts.pkl")

#         if not os.path.exists(output_path):
            
#             print(f"Processing {f}")
            
#             dataset = f.split("/")[-2].replace("_sieved", "")

#             current_path = os.path.join(f, "EK-FAC_scores.pkl")
#             if not os.path.exists(current_path): continue 

#             IF_scores = load_pickle(current_path)["all_modules"]
#             train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
#             test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]
#             current_labels = load_pickle(os.path.join(data_folder, dataset, f"{dataset}_w_mol_info.pkl"))
#             current_labels = {str(r["id_"]): r for r in current_labels}

#             helpful_harmful_percent = {} 

#             for test_idx in tqdm(range(len(test_id_list))):

#                 test_id = test_id_list[test_idx]
#                 current_scores = IF_scores[test_idx, :]

#                 identical_mol_train_iden_expt_cond = get_index_of_identical_mol_identical_expt_cond(current_labels, test_id, train_id_list)
  
#                 if len(identical_mol_train_iden_expt_cond) == 0: continue 

#                 identical_mol_helpful = [i for i in identical_mol_train_iden_expt_cond if current_scores[i] > 0 and abs(current_scores[i]) > threshold]
#                 identical_mol_harmful = [i for i in identical_mol_train_iden_expt_cond if current_scores[i] < 0 and abs(current_scores[i]) > threshold]

#                 num_mol_helpful = len(identical_mol_helpful)
#                 num_mol_harmful = len(identical_mol_harmful)

#                 helpful_harmful_percent[test_id] = {"helpful": num_mol_helpful,
#                                                     "harmful": num_mol_harmful,
#                                                     "total": len(identical_mol_train_iden_expt_cond)}

#             pickle_data(helpful_harmful_percent, output_path)

# def get_identical_mol_MS_sim_harmful_helpful(data_folder, results_folder):

#     results_folder_random = [f for f in results_folder if "random" in f]

#     for f in results_folder_random:
        
#         current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
#         if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
#         output_path = os.path.join(current_output_folder, "identical_mol_MS_sim_helpful_harmful_counts.pkl")
#         dataset = f.split("/")[-2].replace("_sieved", "")
#         current_data_folder = os.path.join(data_folder, dataset, "frags_preds")

#         if not os.path.exists(output_path):
            
#             print(f"Processing {f}")
            
#             current_path = os.path.join(f, "EK-FAC_scores.pkl")
#             if not os.path.exists(current_path): continue 

#             print(current_path)
#             IF_scores = load_pickle(current_path)["all_modules"]
#             train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
#             print(train_id_list[:5])
#             a = z 

#             test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]
#             current_labels = load_pickle(os.path.join(data_folder, dataset, f"{dataset}_w_mol_info.pkl"))
#             current_labels = {str(r["id_"]): r for r in current_labels}

#             helpful_harmful_MS_sim = {} 

#             for test_idx in tqdm(range(len(test_id_list))):

#                 test_id = test_id_list[test_idx]
#                 current_scores = IF_scores[test_idx, :]

#                 identical_mol_train = get_index_of_identical_mol(current_labels, test_id, train_id_list)

#                 if len(identical_mol_train) == 0: continue
#                 identical_mol_helpful = [i for i in identical_mol_train if current_scores[i] > 0]
#                 identical_mol_harmful = [i for i in identical_mol_train if current_scores[i] < 0]

#                 MS_sim_harmful = get_MS_sim(current_data_folder, test_id, identical_mol_harmful)
#                 MS_sim_helpful = get_MS_sim(current_data_folder, test_id, identical_mol_helpful)

#                 helpful_harmful_MS_sim[test_id] = {"helpful": {identical_mol_helpful[i]: MS_sim_helpful[i] for i in range(len(identical_mol_helpful))},
#                                                    "harmful": {identical_mol_harmful[i]: MS_sim_harmful[i] for i in range(len(identical_mol_harmful))}}

#             pickle_data(helpful_harmful_MS_sim, output_path)

# def get_mol_sim_harmful_helpful(data_folder, results_folder, top_k):

#     for f in results_folder:

#         current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
#         if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
#         output_path = os.path.join(current_output_folder, "mol_sim_helpful_harmful.pkl")
        
#         dataset = f.split("/")[-2].replace("_sieved", "")
#         current_data_folder = os.path.join(data_folder, dataset)

#         if not os.path.exists(output_path):
            
#             print(f"Processing {f}")
            
#             current_path = os.path.join(f, "EK-FAC_scores.pkl")
#             if not os.path.exists(current_path): continue 

#             IF_scores = load_pickle(current_path)["all_modules"]
            
#             train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
#             test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]

#             current_labels = load_pickle(os.path.join(current_data_folder, f"{dataset}_w_mol_info.pkl"))
#             current_labels = {str(r["id_"]): r for r in current_labels}

#             helpful_harmful_mol_sim = {} 

#             for test_idx in tqdm(range(len(test_id_list))):

#                 test_id = test_id_list[test_idx]
#                 current_scores = IF_scores[test_idx, :]
#                 _, indices = torch.sort(current_scores)
#                 top_k_harmful, top_k_helpful = indices[:top_k].numpy().tolist(), indices[-top_k:].numpy().tolist()

#                 harmful_train = [train_id_list[i] for i in top_k_harmful]
#                 helpful_train = [train_id_list[i] for i in top_k_helpful]
                
#                 mol_sim_harmful = get_mol_sim(current_labels, test_id, harmful_train)
#                 mol_sim_helpful = get_mol_sim(current_labels, test_id, helpful_train)

#                 helpful_harmful_mol_sim[test_id] = {"helpful": {helpful_train[i]: mol_sim_helpful[i] for i in range(len(helpful_train))},
#                                                     "harmful": {harmful_train[i]: mol_sim_harmful[i] for i in range(len(harmful_train))}}
                
#             # Save the results
#             pickle_data(helpful_harmful_mol_sim, output_path)

# def get_MS_sim_harmful_helpful(data_folder, results_folder, top_k):

#     for f in results_folder:

#         current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
#         if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
#         output_path = os.path.join(current_output_folder, "MS_sim_helpful_harmful.pkl")
        
#         dataset = f.split("/")[-2].replace("_sieved", "")
#         current_data_folder = os.path.join(data_folder, dataset, "frags_preds")

#         if not os.path.exists(output_path):
            
#             print(f"Processing {f}")
            
#             current_path = os.path.join(f, "EK-FAC_scores.pkl")
#             if not os.path.exists(current_path): continue 

#             IF_scores = load_pickle(current_path)["all_modules"]
            
#             train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
#             test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]

#             helpful_harmful_MS_sim = {} 

#             for test_idx in tqdm(range(len(test_id_list))):

#                 test_id = test_id_list[test_idx]
#                 current_scores = IF_scores[test_idx, :]
#                 _, indices = torch.sort(current_scores)
#                 top_k_harmful, top_k_helpful = indices[:top_k].numpy().tolist(), indices[-top_k:].numpy().tolist()

#                 harmful_train = [train_id_list[i] for i in top_k_harmful]
#                 helpful_train = [train_id_list[i] for i in top_k_helpful]
                
#                 MS_sim_harmful = get_MS_sim(current_data_folder, test_id, harmful_train)
#                 MS_sim_helpful = get_MS_sim(current_data_folder, test_id, helpful_train)

#                 helpful_harmful_MS_sim[test_id] = {"helpful": {helpful_train[i]: MS_sim_helpful[i] for i in range(len(helpful_train))},
#                                                    "harmful": {harmful_train[i]: MS_sim_harmful[i] for i in range(len(harmful_train))}}
                
#             # Save the results
#             pickle_data(helpful_harmful_MS_sim, output_path)

# def get_CF_overlap_harmful_helpful(data_folder, results_folder, top_k):

#     for f in results_folder:

#         current_output_folder = os.path.join(IF_cache_folder, f.split("/")[-1])
#         if not os.path.exists(current_output_folder): os.makedirs(current_output_folder)
#         output_path = os.path.join(current_output_folder, "CF_overlap_helpful_harmful.pkl")
        
#         dataset = f.split("/")[-2].replace("_sieved", "")
#         current_data_folder = os.path.join(data_folder, dataset, "frags_preds")

#         if not os.path.exists(output_path):
            
#             print(f"Processing {f}")
            
#             current_path = os.path.join(f, "EK-FAC_scores.pkl")
#             if not os.path.exists(current_path): continue 

#             IF_scores = load_pickle(current_path)["all_modules"]
            
#             train_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "train_ids.pkl"))]
#             test_id_list = [i.replace(".ms", "") for i in load_pickle(os.path.join(f, "test_ids.pkl"))]

#             helpful_harmful_CF_overlap = {} 

#             for test_idx in tqdm(range(len(test_id_list))):

#                 test_id = test_id_list[test_idx]
#                 current_scores = IF_scores[test_idx, :]
#                 _, indices = torch.sort(current_scores)
#                 top_k_harmful, top_k_helpful = indices[:top_k].numpy().tolist(), indices[-top_k:].numpy().tolist()

#                 harmful_train = [train_id_list[i] for i in top_k_harmful]
#                 helpful_train = [train_id_list[i] for i in top_k_helpful]
                
#                 CF_overlap_harmful = get_CF_overlap(current_data_folder, test_id, harmful_train)
#                 CF_overlap_helpful = get_CF_overlap(current_data_folder, test_id, helpful_train)

#                 helpful_harmful_CF_overlap[test_id] = {"helpful": {helpful_train[i]: CF_overlap_helpful[i] for i in range(len(helpful_train))},
#                                                        "harmful": {harmful_train[i]: CF_overlap_harmful[i] for i in range(len(harmful_train))}}
                
#             # Save the results
#             pickle_data(helpful_harmful_CF_overlap, output_path)

def get_index_of_identical_mol(data_folder, results_folder):

    results_folder_random = [f for f in results_folder if "random" in f]

    for folder in results_folder_random:

        expt_name = folder.split("/")[-1]
        output_folder = os.path.join(IF_cache_folder, expt_name)
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "identical_molecules.pkl")
        if os.path.exists(output_path): continue 

        # Get the train and test id list 
        train_id_list_path = os.path.join(folder, "train_ids.pkl")
        test_id_list_path = os.path.join(folder, "test_ids.pkl")

        train_id_list = [str(t).split("/")[-1].replace(".pkl", "").replace(".ms", "") for t in load_pickle(train_id_list_path)]
        test_id_list = [str(t).split("/")[-1].replace(".pkl", "").replace(".ms", "") for t in load_pickle(test_id_list_path)]

        # Get the labels 
        dataset = folder.split("/")[-2].replace("_sieved", "")
        current_labels = load_pickle(os.path.join(data_folder, dataset, f"{dataset}_w_mol_info.pkl"))
        current_labels = {str(r["id_"]): r for r in current_labels}
        
        # For each test id, get the list of spectra that are
        # 1. same molecule, different experimental setting 
        # 2. same molecule, same experimental setting

        iden_mol_iden_expt_list, iden_mol_diff_expt_list = {}, {}

        for test_id in tqdm(test_id_list):
            
            test_info = current_labels[test_id]
            test_mol = test_info["inchikey_original"][:14]
            if test_info["collision_energy"] is None: continue
            if test_info["collision_energy"] == "-": continue

            iden_mol_iden_expt_list[test_id] = []
            iden_mol_diff_expt_list[test_id] = []

            for train_idx, train_id in enumerate(train_id_list):

                train_info = current_labels[train_id]
                train_mol = train_info["inchikey_original"][:14]
                if train_info["collision_energy"] is None: continue
                if train_info["collision_energy"] == "-": continue

                if test_mol != train_mol: continue

                diff_expt_cond = False
                iden_expt_cond = True 

                if test_info["precursor_type"] != train_info["precursor_type"]: 
                    diff_expt_cond = True
                    iden_expt_cond = False
                if test_info["instrument_type"] != train_info["instrument_type"]: 
                    diff_expt_cond = True
                    iden_expt_cond = False

                # Determine how collision energy affects if something is determined as identical experimental condition or not
                if abs(test_info["collision_energy"] - train_info["collision_energy"]) > 5: iden_expt_cond = False
                if abs(test_info["collision_energy"] - train_info["collision_energy"]) > 20: diff_expt_cond = True

                if iden_expt_cond: iden_mol_iden_expt_list[test_id].append((train_idx, train_id))
                if diff_expt_cond: iden_mol_diff_expt_list[test_id].append((train_idx, train_id))

        identical_mol_info = {"identical_expt": iden_mol_iden_expt_list,
                              "different_expt": iden_mol_diff_expt_list}
        
        pickle_data(identical_mol_info, output_path)

def get_identical_mol_percent_harmful_helpful(results_folder):

    results_folder_random = [f for f in results_folder if "random" in f]

    for folder in results_folder_random:

        expt_name = folder.split("/")[-1]
        
        # Set the threshold
        dataset = folder.split("/")[-2].replace("_sieved", "")
        
        if dataset == "canopus": threshold_helpful, threshold_harmful = 5.5, 0.1 #5.5, 0.1
        elif dataset == "massspecgym": threshold_helpful, threshold_harmful = 5.0, 1.0
        elif dataset == "nist2023": threshold_helpful, threshold_harmful = 5.0, 0.5

        identical_molecules_path = os.path.join(IF_cache_folder, expt_name, "identical_molecules.pkl")
        if not os.path.exists(identical_molecules_path): continue

        output_path = os.path.join(IF_cache_folder, expt_name, "identical_molecules_harmful_helpful.pkl")
        # if os.path.exists(output_path): continue
        
        identical_molecules = load_pickle(identical_molecules_path)
        test_id_list_path = os.path.join(folder, "test_ids.pkl")
        test_id_list = [str(t).split("/")[-1].replace(".pkl", "").replace(".ms", "") for t in load_pickle(test_id_list_path)]
        IF_scores = load_pickle(os.path.join(folder, "EK-FAC_scores.pkl"))["all_modules"]

        # Get the helpful and harmful score of identical molecules + same / diff expt
        iden_expt_helpful, iden_expt_harmful = [],[]
        diff_expt_helpful, diff_expt_harmful = [],[]

        for test_id, train_list in identical_molecules["identical_expt"].items():
            
            if len(train_list) == 0: continue              
            test_idx = test_id_list.index(test_id)
            train_idx_list = [r[0] for r in train_list]

            current_scores = IF_scores[test_idx, train_idx_list].numpy()

            helpful_scores = [s for s in current_scores if s > 0 and abs(s) > threshold_helpful]
            harmful_scores = [s for s in current_scores if s < 0 and abs(s) > threshold_harmful]

            if len(helpful_scores) == 0 and len(harmful_scores) == 0: continue 

            iden_expt_helpful.extend(helpful_scores)
            iden_expt_harmful.extend(harmful_scores)
        
        n_iden_expt = len(iden_expt_helpful) + len(iden_expt_harmful)

        for test_id, train_list in identical_molecules["different_expt"].items():
            
            if len(train_list) == 0: continue 
            test_idx = test_id_list.index(test_id)
            train_idx_list = [r[0] for r in train_list]
            current_scores = IF_scores[test_idx, train_idx_list].numpy()
            helpful_scores = [s for s in current_scores if s > 0 and abs(s) > threshold_helpful] 
            harmful_scores = [s for s in current_scores if s < 0 and abs(s) > threshold_harmful]
            
            if len(helpful_scores) == 0 and len(harmful_scores) == 0: continue

            diff_expt_helpful.extend(helpful_scores)
            diff_expt_harmful.extend(harmful_scores)
            
        n_diff_expt = len(diff_expt_helpful) + len(diff_expt_harmful)

        # Get the consolidated statistics
        percent_ident_expt_helpful = len(iden_expt_helpful) / n_iden_expt
        percent_ident_expt_harmful = len(iden_expt_harmful) / n_iden_expt
        percent_diff_expt_helpful = len(diff_expt_helpful) / n_diff_expt
        percent_diff_expt_harmful = len(diff_expt_harmful) / n_diff_expt

        n_expt = n_iden_expt + n_diff_expt
        percent_helpful = (len(diff_expt_helpful) + len(iden_expt_helpful)) / n_expt
        percent_harmful = (len(diff_expt_harmful) + len(iden_expt_harmful)) / n_expt
        
        stats = {}
        stats["same_mol"] = {"helpful": percent_helpful,
                             "harmful": percent_harmful}
        
        stats["same_mol_diff_expt"] = {"helpful": percent_diff_expt_helpful,
                                       "harmful": percent_diff_expt_harmful}
        
        stats["same_mol_ident_expt"] = {"helpful": percent_ident_expt_helpful,
                                        "harmful": percent_ident_expt_harmful}
        
        pprint(stats)
        print()
        pickle_data(stats, output_path)

def get_identical_mol_MS_sim_harmful_helpful(data_folder, results_folder):

    results_folder_random = [f for f in results_folder if "random" in f]

    for folder in results_folder_random:

        expt_name = folder.split("/")[-1]
        dataset = folder.split("/")[-2].replace("_sieved", "")
        identical_molecules_path = os.path.join(IF_cache_folder, expt_name, "identical_molecules.pkl")
        if not os.path.exists(identical_molecules_path): continue

        output_path = os.path.join(IF_cache_folder, expt_name, "identical_molecules_MS_sim_harmful_helpful.pkl")
        # if os.path.exists(output_path): continue
        
        identical_molecules = load_pickle(identical_molecules_path)
        test_id_list_path = os.path.join(folder, "test_ids.pkl")
        test_id_list = [str(t).split("/")[-1].replace(".pkl", "").replace(".ms", "") for t in load_pickle(test_id_list_path)]
        IF_scores = load_pickle(os.path.join(folder, "EK-FAC_scores.pkl"))["all_modules"]

        # Get the helpful and harmful score of identical molecules + same / diff expt
        iden_expt_helpful, iden_expt_harmful = [],[]
        diff_expt_helpful, diff_expt_harmful = [],[]

        for test_id, train_list in identical_molecules["identical_expt"].items():
            
            if len(train_list) == 0: continue 
            test_idx = test_id_list.index(test_id)
            test_info = load_pickle(os.path.join(data_folder, dataset, "frags_preds", test_id + ".pkl"))

            for (train_idx, train_id) in train_list:

                train_info = load_pickle(os.path.join(data_folder, dataset, "frags_preds", train_id + ".pkl"))
                MS_sim = compute_MS_sim(test_info, train_info)
                score = IF_scores[test_idx, train_idx].item()

                if score > 0: iden_expt_helpful.append((score, MS_sim))
                if score < 0: iden_expt_harmful.append((score, MS_sim))
        
        for test_id, train_list in identical_molecules["different_expt"].items():
            
            if len(train_list) == 0: continue 
            test_idx = test_id_list.index(test_id)
            test_info = load_pickle(os.path.join(data_folder, dataset, "frags_preds", test_id + ".pkl"))

            for (train_idx, train_id) in train_list:

                train_info = load_pickle(os.path.join(data_folder, dataset, "frags_preds", train_id + ".pkl"))
                MS_sim = compute_MS_sim(test_info, train_info)
                score = IF_scores[test_idx, train_idx].item()

                if score > 0: diff_expt_helpful.append((score, MS_sim))
                if score < 0: diff_expt_harmful.append((score, MS_sim))
            
        # Get the consolidated statistics
        stats = {"helpful": {"iden_expt": iden_expt_helpful, "diff_expt": diff_expt_helpful}, 
                 "harmful": {"iden_expt": iden_expt_harmful, "diff_expt": diff_expt_harmful}}
        
        pickle_data(stats, output_path)

if __name__ == "__main__":

    cache_folder = "../cache"
    IF_cache_folder = os.path.join(cache_folder, "IF_results")
    if not os.path.exists(IF_cache_folder): os.makedirs(IF_cache_folder)
    
    mist_folder = "../../FP_prediction/mist/best_models"
    baseline_models_folder = "../../FP_prediction/baseline_models/best_models"
    data_folder = "/data/rbg/users/klingmin/projects/MS_processing/data"

    results_folder = []

    for folder in [baseline_models_folder, mist_folder]: # Removed mist for the time being 
        for dataset in os.listdir(folder):
            for checkpoint in os.listdir(os.path.join(folder, dataset)):
                if "nist2023" not in dataset: continue 
                if "sieved" in dataset: continue
                results_folder.append(os.path.join(folder, dataset, checkpoint))

    # 1. Let us get the index of identical molecules (both same and diff expt conditions)
    get_index_of_identical_mol(data_folder, results_folder)

    # # 2. Get percentage of time where identical molecules are harmful / helpful 
    # get_identical_mol_percent_harmful_helpful(results_folder)

    # # 3. Get the MS sim of harmful / helpful records from identical molecules 
    get_identical_mol_MS_sim_harmful_helpful(data_folder, results_folder)
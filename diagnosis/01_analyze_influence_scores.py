import os 
from tqdm import tqdm
from pathlib import Path
from utils import load_pickle, update_dict, same_expt, get_stats, pickle_data

def main(expt_folder, data):

    train_ids_path = expt_folder / "train_ids.pkl"
    test_ids_path = expt_folder / "test_ids.pkl"
    output_path = folder / dataset / expt / "scores_stats.pkl"

    if os.path.exists(output_path): return 
            
    scores = load_pickle(scores_path)["all_modules"]
    train_ids, test_ids = load_pickle(train_ids_path), load_pickle(test_ids_path)
    
    # Create a master list to conslidate the statistics
    master_stats = {"same_mol" : {"helpful": 0, "harmful": 0},
                    "harmful": {"mol_sim": 0, "MS_sim": 0, "count": 0},
                    "helpful": {"mol_sim": 0, "MS_sim": 0, "count": 0},
                    "same_mol_harmful_conditions" : {e: 0 for e in DIFF_EXPT_CONDITIONS},
                    "same_mol_helpful_conditions" : {e: 0 for e in DIFF_EXPT_CONDITIONS},
                    "same_mol_same_conditions": {"helpful": 0, "harmful": 0}}

    # Iterate through now 
    for test_idx, test_id in tqdm(enumerate(test_ids)):

        test_info = data[Path(test_id).stem]
        test_mol = test_info["inchikey_original"]

        train_records = {Path(train_id).stem: update_dict(scores[test_idx, train_idx].item(), data[Path(train_id).stem], "score") for train_idx, train_id in enumerate(train_ids)}

        # Get harmful and helpful molecules 
        harmful = {k:v for k,v in train_records.items() if v["score"] < 0}
        helpful = {k:v for k,v in train_records.items() if v["score"] > 0}
        
        top_k_harmful = {r[0]: r[1] for r in sorted(harmful.items(), key = lambda x: x[1]["score"])[:TOP_K]}
        top_k_helpful = {r[0]: r[1] for r in sorted(helpful.items(), key = lambda x: x[1]["score"], reverse = True)[:TOP_K]}

        # Look at how many harmful are of same / different molecules
        harmful_same = {k:v for k, v in harmful.items() if v["inchikey_original"][:14] == test_mol[:14]}
        helpful_same = {k:v for k, v in helpful.items() if v["inchikey_original"][:14] == test_mol[:14]}

        harmful_diff = {k:v for k, v in harmful.items() if v["inchikey_original"][:14] != test_mol[:14]}
        helpful_diff = {k:v for k, v in helpful.items() if v["inchikey_original"][:14] != test_mol[:14]}
        
        # Get same molecule but with either same or different experimental conditions 
        same_mol_same_conditions = {k: v for k, v in train_records.items() \
                                    if v["inchikey_original"][:14] == test_mol[:14] 
                                    and same_expt(test_info, v)}

        same_mol_diff_conditions = {k: v for k, v in train_records.items() \
                                    if v["inchikey_original"][:14] == test_mol[:14] 
                                    and not same_expt(test_info, v)}
        
        # Look at how many of the same mol with same / diff conditions are helpful / harmful
        harmful_same_mol_same_conditions = {k:v for k,v in same_mol_same_conditions.items() if v["score"] < 0}
        helpful_same_mol_same_conditions = {k:v for k,v in same_mol_same_conditions.items() if v["score"] > 0}

        harmful_same_mol_diff_conditions = {k:v for k,v in same_mol_diff_conditions.items() if v["score"] < 0}
        helpful_same_mol_diff_conditions = {k:v for k,v in same_mol_diff_conditions.items() if v["score"] > 0}

        # Get the statistics now
        harmful_stats = get_stats(test_info, top_k_harmful, compute_mol_dist= True, compute_MS_dist= True)
        helpful_stats = get_stats(test_info, top_k_helpful, compute_mol_dist= True, compute_MS_dist= True)
        
        harmful_same_mol_diff_conditions_stats = get_stats(test_info, harmful_same_mol_diff_conditions) 
        helpful_same_mol_diff_conditions_stats = get_stats(test_info, helpful_same_mol_diff_conditions) 

        # sum everything up now 
        master_stats["same_mol"]["harmful"] += len(harmful_same)
        master_stats["same_mol"]["helpful"] += len(helpful_same)

        master_stats["harmful"]["mol_sim"] += harmful_stats["mol_sim"]
        master_stats["harmful"]["MS_sim"] += harmful_stats["MS_sim"]
        master_stats["harmful"]["count"] += harmful_stats["mol_sim_n_train"]
                        
        master_stats["helpful"]["mol_sim"] += helpful_stats["mol_sim"]
        master_stats["helpful"]["MS_sim"] += helpful_stats["MS_sim"]
        master_stats["helpful"]["count"] += helpful_stats["mol_sim_n_train"]
        
        master_stats["same_mol_same_conditions"]["harmful"] += len(harmful_same_mol_same_conditions)
        master_stats["same_mol_same_conditions"]["helpful"] += len(helpful_same_mol_same_conditions)

        # Add to the conditions now 
        for k,v in harmful_same_mol_diff_conditions_stats["diff_expt_count"].items():
            if k == "total": continue
            master_stats["same_mol_harmful_conditions"][k] += v 

        for k,v in helpful_same_mol_diff_conditions_stats["diff_expt_count"].items():
            if k == "total": continue
            master_stats["same_mol_helpful_conditions"][k] += v 
    
    pickle_data(master_stats, output_path)

if __name__ == "__main__":

    TOP_K = 500
    CACHE_FOLDER = "./cache"
    DATA_FOLDER = Path("/data/rbg/users/klingmin/projects/MS_processing/data")
    RESULTS_FOLDER = ["../FP_prediction/baseline_models/models_cached/w_meta", "../FP_prediction/baseline_models/models_cached/wo_meta"]
    RESULTS_FOLDER = [Path(f) for f in RESULTS_FOLDER]
    if not os.path.exists(CACHE_FOLDER): os.makedirs(CACHE_FOLDER)

    DIFF_EXPT_CONDITIONS = ["diff_adduct", "diff_instrument", "diff_CE",
                            "diff_adduct_instrument", "diff_adduct_CE",
                            "diff_instrument_CE", "diff_adduct_instrument_CE"]

    # Get statistics for each experiment 
    for folder in RESULTS_FOLDER:

        for dataset in os.listdir(folder): 

            data = load_pickle(DATA_FOLDER / dataset / f"{dataset}_w_mol_info.pkl")
            data = {str(rec["id_"]): rec for rec in data}

            for expt in os.listdir(folder / dataset):

                expt_folder = folder / dataset / expt
                scores_path = expt_folder / "EK-FAC_scores.pkl"

                if not os.path.exists(scores_path): continue 

                # Proceed to process the influence scores now 
                main(expt_folder, data)


import os 
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import load_pickle, pickle_data, load_json

def get_CF_statistics(frags_folder):
    
    all_CF_combos = set()
    all_mols = set()
    combo_mol_matching = {} 

    for filename in tqdm(os.listdir(frags_folder)):

        file_path = os.path.join(frags_folder, filename)
        rec = load_pickle(file_path)

        # Get inchikey / list of formula 
        inchikey = rec["inchikey_original"][:14]
        FP_list = [p["comment"]["f_pred"] for p in rec["peaks"]]
        FP_list = sorted([p for p in FP_list if p != ""])
        FP_list = "_".join(FP_list)

        # Update the dictionary 
        if FP_list not in combo_mol_matching:
            combo_mol_matching[FP_list] = [] 

        all_CF_combos.update([FP_list])
        all_mols.update([inchikey])
        combo_mol_matching[FP_list].append((filename.replace(".pkl", ""), inchikey))

    return all_CF_combos, all_mols, combo_mol_matching

if __name__ == "__main__":

    main_folder = Path("/data/rbg/users/klingmin/projects/MS_processing/") 
    data_folder = main_folder / "data"
    MIST_data_folder = main_folder/ "data_BM" / "mist"

    splits_folder = Path("/data/rbg/users/klingmin/projects/MS_processing/data_splits")
    cache_folder = "../cache/CF_unique_combos"
    CFs_folder = Path("/data/rbg/users/klingmin/projects/MS_processing/CFs")
    if not os.path.exists(cache_folder): os.makedirs(cache_folder)

    datasets = ["massspecgym", "nist2023"]

    for dataset in tqdm(datasets): 

        folder = data_folder / dataset
        current_frags_folder = folder / "frags_preds"
        all_CF_combos_output_path = os.path.join(cache_folder, f"{dataset}_all_CFs_combo.pkl")
        all_mols_output_path = os.path.join(cache_folder, f"{dataset}_all_mols_combo.pkl")
        combo_mol_matching_output_path = os.path.join(cache_folder, f"{dataset}_combo_mol_matching.pkl")

        # if os.path.exists(all_CF_combos_output_path) \
        #    and os.path.exists(all_mols_output_path) \
        #    and os.path.exists(combo_mol_matching_output_path): continue 

        print(f"Processing {dataset} now")

        # Get some statistics
        all_CF_combos, all_mols, combo_mol_matching = get_CF_statistics(current_frags_folder)
        pickle_data(all_CF_combos, all_CF_combos_output_path)
        pickle_data(all_mols, all_mols_output_path)
        pickle_data(combo_mol_matching, combo_mol_matching_output_path)
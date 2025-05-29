import os 
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import load_pickle, pickle_data, load_json

def get_CF(data_folder, data_ids):
    
    all_CFs = set()

    for id in tqdm(data_ids):

        data = load_pickle(data_folder / id)
        peaks = data["peaks"]
        f_pred = [f["comment"]["f_pred"] for f in peaks]
        f_pred = [f for f in f_pred if f != ""]
        all_CFs.update(f_pred)
    
    return all_CFs

def get_CF_MIST(current_folder, s):

    current_formula_folder = current_folder / "subformulae" / "default_subformulae"
    current_split_file = current_folder / "splits" / f"{s}.tsv"

    splits = pd.read_csv(current_split_file, sep = "\t").to_dict()
    id_list, split_list = splits["name"], splits["split"]

    train_CFs, val_CFs, test_CFs = set(), set(), set()

    for idx, id_ in tqdm(id_list.items()):
        
        try:
            split = split_list[idx].strip()
            formula = load_json(current_formula_folder / f"{id_}.json")["output_tbl"]["formula"]

            if split == "train": train_CFs.update(formula)
            elif split == "val": val_CFs.update(formula)
            elif split == "test": test_CFs.update(formula)
            else: raise Exception(f"{split} is unknown")
        
        except Exception as e:
            print(e)

    return train_CFs, val_CFs, test_CFs

if __name__ == "__main__":

    main_folder = Path("/data/rbg/users/klingmin/projects/MS_processing/") 
    data_folder = main_folder / "data"
    MIST_data_folder = main_folder/ "data_BM" / "mist"

    splits_folder = Path("/data/rbg/users/klingmin/projects/MS_processing/data_splits")
    CFs_folder = Path("/data/rbg/users/klingmin/projects/MS_processing/CFs")
    if not os.path.exists(CFs_folder): os.makedirs(CFs_folder)

    datasets = ["massspecgym", "nist2023"]
    considered_splits = ["random_sieved", "scaffold_vanilla_sieved", "inchikey_vanilla_sieved", "LS_sieved"]

for dataset in tqdm(datasets): 

    folder = data_folder / dataset
    frags_folder = folder / "frags_preds"
    splits = splits_folder / dataset / "splits"

    for s in tqdm(considered_splits):

        current_split = load_json(splits / f"{s}.json")
        CFs = CFs_folder / dataset/ f"{s}_split"
        if not os.path.exists(CFs): os.makedirs(CFs)

        # 1. For the baseline models that we have designed
        train_path = CFs / "train_CFs.pkl"
        val_path = CFs / "val_CFs.pkl"
        test_path = CFs / "test_CFs.pkl"
        
        if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):

            print(f"Processing {dataset} - {s} split now")

            train_ids, val_ids, test_ids = current_split["train"], current_split["val"], current_split["test"]

            if not os.path.exists(train_path):
                train_CFs = get_CF(frags_folder, train_ids)
                pickle_data(train_CFs, train_path)

            if not os.path.exists(val_path):
                val_CFs = get_CF(frags_folder, val_ids)
                pickle_data(val_CFs, val_path)

            if not os.path.exists(test_path):
                test_CFs = get_CF(frags_folder, test_ids)
                pickle_data(test_CFs, test_path)
        
        # 2. For the MIST model 
        train_MIST_path = CFs / "train_MIST_CFs.pkl"
        val_MIST_path = CFs / "val_MIST_CFs.pkl"
        test_MIST_path = CFs / "test_MIST_CFs.pkl"
        
        if not os.path.exists(train_MIST_path) or not os.path.exists(val_MIST_path) or not os.path.exists(test_MIST_path):
            
            print(f"Processing {dataset} - {s} split for MIST now")
            current_folder = MIST_data_folder / dataset
            train_CFs, val_CFs, test_CFs = get_CF_MIST(current_folder, s)

            print(len(train_CFs), len(val_CFs), len(test_CFs))

            pickle_data(train_CFs, train_MIST_path)
            pickle_data(val_CFs, val_MIST_path)
            pickle_data(test_CFs, test_MIST_path)
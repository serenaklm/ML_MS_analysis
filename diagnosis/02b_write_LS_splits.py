import os 
import random 

from utils import load_pickle, write_json


if __name__ == "__main__":

    results_folder = "../FP_prediction/mist/best_ls_results"
    output_folder = "/data/rbg/users/klingmin/projects/MS_processing/data_splits"

    for dataset in ["massspecgym_sieved", "nist2023_sieved"]:

        splits_folder = os.path.join(results_folder, f"MIST_{dataset}")

        if "sieved" in dataset:
            dataset = dataset.replace("_sieved", "")
            current_output_folder = os.path.join(output_folder, dataset, "splits")
            current_output_path = os.path.join(current_output_folder, "LS_sieved.json")
        else:
            current_output_folder = os.path.join(output_folder, dataset, "splits")
            current_output_path = os.path.join(current_output_folder, "LS.json")

        # if os.path.exists(current_output_path): continue 

        data_ids = load_pickle(os.path.join(splits_folder, "data_ids.pkl"))
        best_split = load_pickle(os.path.join(splits_folder, "best_split.pkl"))

        train_idx, test_idx = best_split["train_indices"], best_split["test_indices"]
        train_ids = [data_ids[i] for i in train_idx]
        test_ids = [data_ids[i] for i in test_idx]

        random.shuffle(train_ids)
        n_val = int(0.2 * len(train_ids))
        val_ids = train_ids[:n_val]
        train_ids = train_ids[n_val:]

        print(f"There are {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        split = {"train": [i + ".pkl" for i in train_ids],
                 "val": [i + ".pkl" for i in val_ids],
                 "test": [i + ".pkl" for i in test_ids]}

        write_json(split, current_output_path)
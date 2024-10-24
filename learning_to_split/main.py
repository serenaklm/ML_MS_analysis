import os 
import copy
import argparse 
from datetime import datetime
from typing import Any, Callable, List

from torch.utils.data import Dataset, Subset
from pytorch_lightning import seed_everything


from utils import read_config, get_all_spectra

from training import split_data
from models.build import ModelFactory



def learning_to_split(config: dict, 
                      data: List,
                      verbose: bool = True):
    
    # Set a random seed 
    seed_everything(config["seed"])

    num_no_improvements = 0
    best_gap, best_split = -1, None

    # Sanity check 
    train_ratio = config["model"]["train_params"]["train_ratio"]
    assert train_ratio > 0.0 and train_ratio < 1.0, "Training ratio needs to be between 0.0 and 1.0."

    splitter = ModelFactory.get_model(config, splitter = True)

    for outer_loop in range(config["model"]["train_params"]["n_outer_loops"]):

        predictor = ModelFactory.get_model(config, predictor = True)

        random_split = True if outer_loop == 0 else False
        split_stats, train_indices, test_indices = split_data(data, splitter, config, random_split) 

        print("okay i can reach this point")
        a = z  
        # val_score = train_predictor(data = data, train_indices = train_indices,
        #                             predictor = predictor, args = args)
        # test_score = test_predictor(data = data, test_indices = test_indices,
        #                             predictor = predictor, args = args)
        
        # if verbose: print_split_status(outer_loop, split_stats, val_score, test_score)



if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description = "Trying to find difficult splits")

    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "base_config.yaml", help = "Config file")

    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", help = "Enable wandb logging")
    parser.add_argument("--user", type = str, default = "serenakhoolm", help = "Set the user")

    args = parser.parse_args()

    # Read in the config, data, then train 
    config = read_config(os.path.join(args.config_dir, args.config_file))
    train_path = os.path.join(config["data"]["dir"], "train.msp")
    val_path = os.path.join(config["data"]["dir"], "val.msp")
    test_path = os.path.join(config["data"]["dir"], "test.msp")

    data = get_all_spectra(train_path) + get_all_spectra(val_path) + get_all_spectra(test_path)
    print(f"{len(data)} records in total.")

    learning_to_split(config, data)

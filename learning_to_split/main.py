import os 
import copy
import argparse 
from datetime import datetime
from typing import Any, Callable, List

from torch.utils.data import Dataset, Subset

from utils import read_config, get_all_spectra

from dataloader import CustomedDataset
from models.build import ModelFactory
from training import split_data, train_predictor

def update_config(config):
    
    # # Update the input_dim 
    input_dim = int(config["dataloader"]["max_da"] / config["dataloader"]["bin_resolution"])

    # Update the output_dim 
    FP_dim_mapping = {"maccs": 167,
                      "morgan4_256": 256,
                      "morgan4_1024": 1024, 
                      "morgan4_2048": 2048,
                      "morgan4_4096": 4096,
                      "morgan6_256": 256,
                      "morgan6_1024": 1024, 
                      "morgan6_2048": 2048,
                      "morgan6_4096": 4096}

    if config["dataloader"]["FP_type"] not in FP_dim_mapping: raise ValueError(f"FP type selected not supported.")

    model_name = config["model"]["name"]

    config["model"][model_name]["input_dim"] = input_dim
    config["model"][model_name]["output_dim"] = FP_dim_mapping[config["dataloader"]["FP_type"]]

    return config 

def learning_to_split(config: dict, 
                      data: List,
                      verbose: bool = True):

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

        val_score = train_predictor(data = data, train_indices = train_indices,
                                    predictor = predictor, config = config)
        
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
    config = update_config(config)

    train_path = os.path.join(config["data"]["dir"], "train.msp")
    val_path = os.path.join(config["data"]["dir"], "val.msp")
    test_path = os.path.join(config["data"]["dir"], "test.msp")

    data = get_all_spectra(train_path) + get_all_spectra(val_path) + get_all_spectra(test_path)
    data = CustomedDataset(data, **config["dataloader"])
    
    learning_to_split(config, data)

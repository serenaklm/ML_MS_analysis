import os 
import copy
import argparse 
from datetime import datetime
from typing import Any, Callable, List

import torch
import torch.nn as nn 

from utils import set_seed, read_config, get_all_spectra, print_split_status, get_optim, write_json

from dataloader import CustomedDataset
from models.build import ModelFactory
from training import split_data, train_predictor, test_predictor, train_splitter

def update_config(config, args):
    
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

    # Update the output directory 
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, current_time)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    config["output_dir"] = output_dir

    return config 

def save(splitter: nn.Module, 
         predictor: nn.Module, 
         best_split: dict, 
         output_dir: str):
    
    # Save the stat dict of the splitter and predictor 
    torch.save(splitter.state_dict(), os.path.join(output_dir, "splitter_weights.pth"))
    torch.save(predictor.state_dict(), os.path.join(output_dir, "predictor_weights.pth"))
    write_json(best_split, os.path.join(output_dir, "best_split.json"))

def learning_to_split(config: dict, 
                      data: List,
                      verbose: bool = True):

    num_no_improvements = 0
    best_gap, best_split = -1, None

    # Sanity check 
    train_ratio = config["model"]["train_params"]["train_ratio"]
    assert train_ratio > 0.0 and train_ratio < 1.0, "Training ratio needs to be between 0.0 and 1.0."

    # Get the splitter
    splitter = ModelFactory.get_model(config, splitter = True)
    opt = get_optim(splitter, config)

    for outer_loop in range(config["model"]["train_params"]["n_outer_loops"]):

        # Train the predictor 
        predictor = ModelFactory.get_model(config, predictor = True)
        random_split = True if outer_loop == 0 else False

        split_stats, train_indices, test_indices = split_data(data, splitter, config, random_split) 

        val_score = train_predictor(data = data, train_indices = train_indices,
                                    predictor = predictor, config = config)
        
        test_score = test_predictor(data = data, test_indices = test_indices,
                                    predictor = predictor, config = config)
        
        if verbose: print_split_status(outer_loop, split_stats, val_score, test_score)

        gap = val_score - test_score

        if gap > best_gap:
            
            best_gap, num_no_improvements = gap, 0

            best_split = {"train_indices":  train_indices,
                          "test_indices":   test_indices,
                          "val_score":      val_score,
                          "test_score":     test_score,
                          "split_stats":    split_stats,
                          "outer_loop":     outer_loop,
                          "best_gap":       best_gap}
            
            save(splitter, predictor, best_split, config["output_dir"])

        else: num_no_improvements += 1
        
        if num_no_improvements == config["model"]["train_params"]["patience"]: break

        # Train the splitter
        train_splitter(splitter, predictor, data, test_indices, opt, config,
                       verbose = verbose)

    # Done! Print the best split.
    if verbose:
        print("Finished!\nBest split:")
        print_split_status(best_split["outer_loop"], best_split["split_stats"],
                           best_split["val_score"], best_split["test_score"])

if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description = "Trying to find difficult splits")

    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "base_config.yaml", help = "Config file")
    parser.add_argument("--output_dir", type = str, default = "./results", help = "Output directory")

    args = parser.parse_args()

    # Read in the config, data, then train 
    config = read_config(os.path.join(args.config_dir, args.config_file))
    config = update_config(config, args)

    # Set seed
    set_seed(config["seed"])

    train_path = os.path.join(config["data"]["dir"], "train.msp")
    val_path = os.path.join(config["data"]["dir"], "val.msp")
    test_path = os.path.join(config["data"]["dir"], "test.msp")

    data = get_all_spectra(train_path) + get_all_spectra(val_path) + get_all_spectra(test_path)
    data = CustomedDataset(data, **config["dataloader"])
    
    learning_to_split(config, data)
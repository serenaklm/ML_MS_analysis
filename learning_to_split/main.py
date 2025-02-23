import os 
import copy
import wandb
import yaml
import argparse 
from typing import List
from datetime import datetime

import torch
import torch.nn as nn 

from utils import set_seed, read_config, print_split_status, get_optim, write_json

from dataloader import MSDataset
from models.build import ModelFactory
from training import split_data, train_predictor, test_predictor, train_splitter

def create_results_dir(results_dir):
    if not os.path.exists(results_dir): os.makedirs(results_dir)

def write_config(wandb_logger, config):

    # Dump raw config now
    run_out_dir = wandb_logger.experiment.dir
    config_out_path = os.path.join(run_out_dir, "run.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)
    wandb_logger.experiment.save("run.yaml", policy = "now")

def update_config(config):
    
    # Get the FP_dim_mapping
    FP_dim_mapping = {"MACCS": 167,
                      "morgan4_256": 256,
                      "morgan4_1024": 1024, 
                      "morgan4_2048": 2048,
                      "morgan4_4096": 4096,
                      "morgan6_256": 256,
                      "morgan6_1024": 1024, 
                      "morgan6_2048": 2048,
                      "morgan6_4096": 4096}

    if config["data"]["FP_type"] not in FP_dim_mapping: raise ValueError(f"FP type selected not supported.")

    # Update config for formula_encoder
    config["model"]["formula_encoder"] = copy.deepcopy(config["model"]["MS_encoder"])
    config["model"]["formula_encoder"]["n_atoms"] = len(config["data"]["considered_atoms"])

    # Update config for frag_encoder
    config["model"]["frag_encoder"] = copy.deepcopy(config["model"]["MS_encoder"])
    config["model"]["frag_encoder"]["chemberta_model"] = config["data"]["chemberta_model"]

    # Update params for all models 
    all_models = ["binned_MS_encoder", "MS_encoder", "formula_encoder", "frag_encoder"]
    for m in all_models:

        # Update the input_dim 
        input_dim = int(config["data"]["max_da"] / config["data"]["bin_resolution"])
        config["model"][m]["input_dim"] = input_dim

        # Update the output_dim
        config["model"][m]["output_dim"] = FP_dim_mapping[config["data"]["FP_type"]]

        # Update getting the CF, fragments etc 
        config["data"]["get_CF"] = config["data"]["get_frags"] = False  
    
    # Update getting the CF, fragments for each individual model 
    if config["model"]["name"] == "formula_encoder": config["data"]["get_CF"] = True 
    if config["model"]["name"] == "frag_encoder": config["data"]["get_frags"] = True 

    # # Update the output directory 
    # current_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
    # output_dir = args.output_dir
    # output_dir = os.path.join(output_dir, current_time)
    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # config["output_dir"] = output_dir

    # Update the device 
    device_idx = config["device"]
    if torch.cuda.is_available(): 
        device = torch.device(f"cuda:{device_idx}")
    else:
        device = torch.device("cpu")

    config[device] = device 
    config["model"]["train_params"]["device"] = device 

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

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], config["model"]["name"])
    expt_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

    # Initialize the logger
    wandb.init(project = config["project"],
                         config = config,
                         group = config["args"]["config_file"].replace(".yaml", ""),
                         entity = config["args"]["user"],
                         name = expt_name)
    
    # Dump config
    raw_config = copy.deepcopy(config)
    del raw_config["args"]
    write_config(raw_config, results_dir)

    num_no_improvements = 0
    best_gap, best_split = -1, None

    # Sanity check 
    train_ratio = config["model"]["train_params"]["train_ratio"]
    assert train_ratio > 0.0 and train_ratio < 1.0, "Training ratio needs to be between 0.0 and 1.0."

    print("okay i am here")
    a = z 

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
        wandb_logger.log({"gap/gap": gap})

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
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--results_dir", type = str, default = "./results", help = "Results output directory")
    parser.add_argument("--wandb", action = "store_true", default = True, help = "Enable wandb logging")
    parser.add_argument("--user", type = str, default = "serenakhoolm", help = "Set the user")

    args = parser.parse_args()

    # Set torch hub cache
    torch.hub.set_dir(args.torch_hub_cache)

    # Read in the config, data, then train 
    config = read_config(os.path.join(args.config_dir, args.config_file))
    config["args"] = args.__dict__
    config = update_config(config, args)

    # Set seed
    set_seed(config["seed"])

    # Get the data 
    data = MSDataset(**config["data"])
    
    # Train the model now 
    learning_to_split(config, data)
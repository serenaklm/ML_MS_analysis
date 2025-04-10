import os
import yaml
import copy
import wandb
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from dataloader import MSDataset, Data
from utils import read_config, load_pickle, pickle_data, write_json
from modules import MSBinnedModel, MSTransformerEncoder, FormulaTransformerEncoder

from learning_to_split import set_seed, get_optim, split_data, train_splitter

def update_config(args, config):
    
    config["args"] = args.__dict__
    devices = config["trainer"].get("devices", 1)

    if config["args"]["debug"]:
        torch.autograd.set_detect_anomaly(True)
        devices = 1
        config["trainer"].update(devices=devices)
        config["data"]["num_workers"] = 0

    config["trainer"]["val_check_interval"] = config["trainer"]["log_every_n_steps"] - 1

    if devices > 1:
        config.setdefault("trainer", {}).update(strategy = DDPStrategy(find_unused_parameters=True))

    if args.disable_checkpoint:
        config["trainer"]["enable_checkpointing"] = False

    # Update the data directory 
    config["data"]["dir"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "frags_preds")
    config["data"]["split_file"] = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], "splits", config["data"]["split_file"])
    config["data"]["adduct_file"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "all_adducts.pkl")
    config["data"]["instrument_file"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "all_instruments.pkl")
    del config["data"]["dataset"]
    del config["data"]["data_folder"]
    del config["data"]["splits_folder"]

    # Read in the adduct and instrument file 
    n_adducts = len(load_pickle(config["data"]["adduct_file"]))
    n_CEs = 10 
    n_instruments = len(load_pickle(config["data"]["instrument_file"]))

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

    # Update params for all models 
    all_models = ["binned_MS_encoder", "MS_encoder", "formula_encoder"]
    for m in all_models:

        # Update the input_dim 
        input_dim = int(config["data"]["max_da"] / config["data"]["bin_resolution"])
        config["model"][m]["input_dim"] = input_dim
            
        # Update the positive weight
        pos_weight = int(config["train_params"]["pos_weight"])
        config["model"][m]["pos_weight"] = pos_weight

        # Update the reconstruction weight 
        reconstruction_weight = float(config["train_params"]["reconstruction_weight"]) 
        config["model"][m]["reconstruction_weight"] = reconstruction_weight

        # Update the output_dim
        config["model"][m]["output_dim"] = FP_dim_mapping[config["data"]["FP_type"]]

        # Update getting the CF, fragments etc 
        config["data"]["get_CF"] = config["data"]["get_frags"] = False

        # Update including the energy, adduct and instrument 
        config["model"][m]["include_adduct"] = config["model"]["feats_params"]["include_adduct"]  
        config["model"][m]["include_CE"] = config["model"]["feats_params"]["include_CE"]  
        config["model"][m]["include_instrument"] = config["model"]["feats_params"]["include_instrument"]
        config["model"][m]["n_adducts"] = n_adducts
        config["model"][m]["n_CEs"] = n_CEs
        config["model"][m]["n_instruments"] = n_instruments

    # Update getting the CF, fragments for each individual model 
    if config["model"]["name"] == "formula_encoder": config["data"]["get_CF"] = True 

    return config

@rank_zero_only
def write_config_local(config, config_out_path):
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)

@rank_zero_only
def create_results_dir(results_dir):
    if not os.path.exists(results_dir): os.makedirs(results_dir)

def get_model(config):

    # Get the model
    model_name = config["model"]["name"]
    if model_name == "binned_MS_encoder":
        model = MSBinnedModel(**config["model"]["binned_MS_encoder"], lr = config["train_params"]["learning_rate"], 
                                                                      weight_decay = config["train_params"]["weight_decay"])
    elif model_name == "MS_encoder":
        model = MSTransformerEncoder(**config["model"]["MS_encoder"], lr = config["train_params"]["learning_rate"], 
                                                                      weight_decay = config["train_params"]["weight_decay"])
    elif model_name == "formula_encoder":
        model = FormulaTransformerEncoder(**config["model"]["formula_encoder"], lr = config["train_params"]["learning_rate"], 
                                                                                weight_decay = config["train_params"]["weight_decay"])
    else:
        raise Exception(f"{model_name} not supported.")
    
    return model

def learning_to_split(config: dict, 
                      verbose: bool = True):
    
    # Get the dataset 
    datamodule = MSDataset(**config["data"])
    dataset = Data(datamodule.data, datamodule.process)

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], config["model"]["name"])
    expt_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

    # Write the config here
    config_o = read_config(os.path.join(config["args"]["config_dir"], config["args"]["config_file"]))
    config_o["exp_name"] = expt_name
    write_config_local(config_o, os.path.join(results_dir, "run.yaml"))

    num_no_improvements = 0
    best_gap, best_split = -1, None
    
    # Sanity check 
    train_ratio = config["splitter"]["train_ratio"]
    assert train_ratio > 0.0 and train_ratio < 1.0, "Training ratio needs to be between 0.0 and 1.0."
    
    # Get the splitter
    config_splitter = copy.deepcopy(config)
    for model in ["binned_MS_encoder", "MS_encoder", "formula_encoder"]: 
        config_splitter["model"][model]["output_dim"] = 2 # train and test split 

    splitter = get_model(config_splitter)
    opt = get_optim(splitter, config)

    # Get the wandb logger 
    wandb_logger = WandbLogger(project = config["project"],
                        config = config,
                        group = config["args"]["config_file"].replace(".yaml", ""),
                        entity = config["args"]["user"],
                        name = expt_name,
                        log_model = False)

    # Start training here
    for outer_loop in range(config["train_params"]["n_outer_loops"]):

        predictor = get_model(config)
        
        # Split the data 
        random_split = True if outer_loop == 0 else False
        split_stats, train_indices, test_indices = split_data(dataset, splitter, 
                                                            config["splitter"]["train_ratio"], 
                                                            config["data"]["batch_size"], 
                                                            config["data"]["num_workers"],
                                                            random_split) 

        # Get trainer and logger
        monitor = config["callbacks"]["monitor"]
        earlystop_callback = EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"])
        trainer = pl.Trainer(**config["trainer"], logger = wandb_logger, callbacks=[earlystop_callback])

        # Start the training now
        trainer.fit(predictor, datamodule = datamodule)

        # Get the gap 
        val_loss = trainer.validate(model = predictor, dataloaders = datamodule)[0] #["val_loss"]
        test_loss = trainer.test(model = predictor, dataloaders = datamodule)[0]# ["test_loss"]
        print(val_loss.keys(), test_loss.keys())
        a = z 

        gap = test_loss - val_loss

        if gap > best_gap:
            
            best_gap, num_no_improvements = gap, 0

            best_split = {"train_indices":  train_indices,
                        "test_indices":   test_indices,
                        "val_loss":       val_loss,
                        "test_loss":      test_loss,
                        "split_stats":    split_stats,
                        "outer_loop":     outer_loop,
                        "best_gap":       best_gap}
            
            pickle_data(best_split, os.path.join(results_dir, "best_split.pkl"))
            write_json(split_stats, os.path.join(results_dir, "splits_stats.json"))

        else: num_no_improvements += 1
        if num_no_improvements == config["splitter"]["patience"]: break

        # Train the splitter
        train_splitter(splitter, predictor, dataset, test_indices, opt,
                    config["data"]["batch_size"], config["splitter"]["num_batches"], config["data"]["num_workers"],
                    config, verbose = verbose)
            
    # Done! Print the best split.
    if verbose:
        print("Finished!\nBest split:")
        print(best_split["outer_loop"], best_split["split_stats"],
                        best_split["val_score"], best_split["test_score"])
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "LS_config.yaml", help = "Config file")
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--results_dir", type = str, default = "./ls_results", help = "Results output directory")
    parser.add_argument("--debug", action = "store_true", default = False, help = "Set debug mode")
    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", default = True, help = "Enable wandb logging")
    parser.add_argument("--user", type = str, default = "serenakhoolm", help = "Set the user")

    args = parser.parse_args()

    # Read in the config, data, then start to find the worst split
    config = read_config(os.path.join(args.config_dir, args.config_file))
    config = update_config(args, config)

    # Set seed
    set_seed(config["seed"])

    # Run learning to split now 
    learning_to_split(config)
import os
import yaml
import copy
import shutil
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from utils import read_config
from dataloader import MSDataset
from modules import MSBinnedModel, MSTransformerEncoder, FormulaTransformerEncoder, FragTransformerEncoder

os.environ['TRANSFORMERS_CACHE'] = "/data/rbg/users/klingmin/transformers_cache"

@rank_zero_only
def write_config(wandb_logger, config):

    # Dump raw config now
    run_out_dir = wandb_logger.experiment.dir
    config_out_path = os.path.join(run_out_dir, "run.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)
    wandb_logger.experiment.save("run.yaml", policy = "now")

@rank_zero_only
def write_config_local(config, config_out_path):

    with open(config_out_path, "w") as f:
        yaml.dump(config, f)

@rank_zero_only
def create_results_dir(results_dir):

    if not os.path.exists(results_dir): os.makedirs(results_dir)

def update_config(config):
    
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
            
        # Update the positive weight
        pos_weight = int(config["model"]["train_params"]["pos_weight"])
        config["model"][m]["pos_weight"] = pos_weight

        # Update the reconstruction weight 
        reconstruction_weight = float(config["model"]["train_params"]["reconstruction_weight"]) 
        config["model"][m]["reconstruction_weight"] = reconstruction_weight

        # Update the output_dim
        config["model"][m]["output_dim"] = FP_dim_mapping[config["data"]["FP_type"]]

        # Update getting the CF, fragments etc 
        config["data"]["get_CF"] = config["data"]["get_frags"] = False  
    
    # Update getting the CF, fragments for each individual model 
    if config["model"]["name"] == "formula_encoder": config["data"]["get_CF"] = True 
    if config["model"]["name"] == "frag_encoder": config["data"]["get_frags"] = True 

    return config

def train(config):  

    # Set a random seed 
    seed_everything(config["seed"])

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], config["model"]["name"])
    expt_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

    # Get the wandb logger 
    wandb_logger = None 
    if not config["args"]["debug"] and config["args"]["wandb"]:
        wandb_logger = WandbLogger(project = config["project"],
                                   config = config,
                                   group = config["args"]["config_file"].replace(".yaml", ""),
                                   entity = config["args"]["user"],
                                   name = expt_name,
                                   log_model = False)
        
        # Dump config
        raw_config = copy.deepcopy(config)
        del raw_config["args"]
        write_config(wandb_logger, raw_config)

    # Write the config here
    config_o = read_config(os.path.join(config["args"]["config_dir"], config["args"]["config_file"]))
    config_o["exp_name"] = expt_name
    write_config_local(config_o, os.path.join(results_dir, "run.yaml"))

    # Set output folder
    config["trainer"].update(default_root_dir = results_dir)

    # Get dataset
    dataset = MSDataset(**config["data"])
    dataset.prepare_data()
    dataset.setup()

    # Get trainer and logger
    checkpoint_callback = ModelCheckpoint(monitor="val_average_loss",
                                          dirpath = results_dir,
                                          filename = '{epoch:02d}-{val_average_loss:.3f}',
                                          every_n_train_steps = config["trainer"]["log_every_n_steps"], 
                                          save_top_k = 2, mode = "min")

    trainer = pl.Trainer(**config["trainer"], logger = wandb_logger, callbacks=[checkpoint_callback])

    # Get the model
    model_name = config["model"]["name"]
    if model_name == "binned_MS_encoder":
        model = MSBinnedModel(**config["model"]["binned_MS_encoder"], lr = config["model"]["train_params"]["lr"], 
                                                                      weight_decay = config["model"]["train_params"]["weight_decay"])
    elif model_name == "MS_encoder":
        model = MSTransformerEncoder(**config["model"]["MS_encoder"], lr = config["model"]["train_params"]["lr"], 
                                                                      weight_decay = config["model"]["train_params"]["weight_decay"])
    elif model_name == "formula_encoder":
        model = FormulaTransformerEncoder(**config["model"]["formula_encoder"], lr = config["model"]["train_params"]["lr"], 
                                                                                weight_decay = config["model"]["train_params"]["weight_decay"])
    elif model_name == "frag_encoder":
        model = FragTransformerEncoder(**config["model"]["frag_encoder"], lr = config["model"]["train_params"]["lr"], 
                                                                          weight_decay = config["model"]["train_params"]["weight_decay"])    
    else:
        raise Exception(f"{model_name} not supported.")
    
    # Train     
    trainer.fit(model, datamodule = dataset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "base_config.yaml", help = "Config file")
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--results_dir", type = str, default = "./results", help = "Results output directory")
    parser.add_argument("--debug", action = "store_true", default = False, help = "Set debug mode")
    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", default = True, help = "Enable wandb logging")
    parser.add_argument("--user", type = str, default = "serenakhoolm", help = "Set the user")

    args = parser.parse_args()

    # Set torch hub cache
    torch.hub.set_dir(args.torch_hub_cache)

    # Read in the config
    config = read_config(os.path.join(args.config_dir, args.config_file))
    if "data" not in config: raise ValueError("Missing required key: data")
    if "model" not in config: raise ValueError("Missing required key: model")
    if "trainer" not in config: raise ValueError("Missing required key: trainer")

    # Update the trainer config
    config["args"] = args.__dict__
    config = update_config(config)

    # Run the trainer now 
    train(config)
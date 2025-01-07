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
from modules import MSBinnedModel, MSTransformerEncoder

@rank_zero_only
def write_config(wandb_logger, config):

    # Dump raw config now
    run_out_dir = wandb_logger.experiment.dir
    config_out_path = os.path.join(run_out_dir, "run.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)
    wandb_logger.experiment.save("run.yaml", policy="now")

@rank_zero_only
def write_config_local(config, config_out_path):

    with open(config_out_path, "w") as f:
        yaml.dump(config, f)

def update_config(config):
    
    devices = config["trainer"].get("devices", 1)

    if config["args"]["debug"]:
        torch.autograd.set_detect_anomaly(True)
        devices = 1
        config["trainer"].update(devices=devices)
        config["data"]["num_workers"] = 0

    if devices > 1:
        config.setdefault("trainer", {}).update(strategy = DDPStrategy(find_unused_parameters=False))

    if args.disable_checkpoint:
        config["trainer"]["enable_checkpointing"] = False
    
    # Set output folder
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
    results_dir = os.path.join(config["args"]["results_dir"], config["model"]["name"], current_time)
    config["args"]["results_dir"] = results_dir
    config["trainer"].update(default_root_dir = config["args"]["results_dir"])

    # Update the input_dim 
    input_dim = int(config["data"]["max_da"] / config["data"]["bin_resolution"])
    config["model"]["binned_MS_encoder"]["input_dim"] = input_dim
    config["model"]["MS_encoder"]["input_dim"] = input_dim

    # Update the output_dim 
    FP_dim_mapping = {"morgan4_256": 256,
                      "morgan4_1024": 1024, 
                      "morgan4_2048": 2048,
                      "morgan4_4096": 4096,
                      "morgan6_256": 256,
                      "morgan6_1024": 1024, 
                      "morgan6_2048": 2048,
                      "morgan6_4096": 4096}

    if config["data"]["FP_type"] not in FP_dim_mapping: raise ValueError(f"FP type selected not supported.")
    config["model"]["binned_MS_encoder"]["input_dim"] = input_dim
    config["model"]["binned_MS_encoder"]["output_dim"] = FP_dim_mapping[config["data"]["FP_type"]]

    return config

def train(config):  

    # Set a random seed 
    seed_everything(config["seed"])

    # Write the config here
    config_o = read_config(os.path.join(config["args"]["config_dir"], config["args"]["config_file"]))
    expt_name = config["args"]["results_dir"].split("/")[-1]
    config_o["exp_name"] = expt_name
    write_config_local(config_o, os.path.join(config["args"]["results_dir"], "run.yaml"))

    # Get the wandb logger 
    wandb_logger = None 
    if not config["args"]["debug"] and config["args"]["wandb"]:
        wandb_logger = WandbLogger(project = config["project"],
                                   config = config,
                                   name = expt_name,
                                   group = config["args"]["config_file"].replace(".yaml", ""),
                                   entity = config["args"]["user"],
                                   log_model = False)
        
        # Dump config
        raw_config = copy.deepcopy(config)
        del raw_config["args"]
        write_config(wandb_logger, raw_config)
        
    # Get dataset
    dataset = MSDataset(**config["data"])
    dataset.prepare_data()
    dataset.setup()

    # Get trainer and logger
    checkpoint_callback = ModelCheckpoint(monitor="val/loss",
                                          dirpath = config["args"]["results_dir"],
                                          filename = '{epoch:02d}-{val/loss:.2f}',
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

    # Get the results folder 
    if not os.path.exists(config["args"]["results_dir"]): os.makedirs(config["args"]["results_dir"])

    # Run the trainer now 
    train(config)
import os
import copy
import yaml
import logging
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch 
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import ms_pred.common as common
from ms_pred.dag_pred import dag_data

from model.inten_model import IntenGNN

from utils import read_config

@rank_zero_only
def create_results_dir(results_dir):
    if not os.path.exists(results_dir): os.makedirs(results_dir)

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

def get_dataloaders(config):

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = config["dataset"]["dataset_name"]
    data_dir = Path(config["dataset"]["data_dir"])
    labels = data_dir / dataset_name / config["dataset"]["dataset_labels"]
    split_file = data_dir / dataset_name/ "splits" / config["dataset"]["split_name"]
    add_hs = config["process_setting"]["add_hs"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)

    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    magma_dag_folder = data_dir / dataset_name / Path(config["dataset"]["magma_dag_folder"])
    num_workers = config["process_setting"].get("num_workers", 0)
    all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    name_to_json = {i.stem: i for i in all_json_pths}

    pe_embed_k = config["process_setting"]["pe_embed_k"]
    root_encode = config["process_setting"]["root_encode"]
    binned_targs = config["process_setting"]["binned_targs"]
    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k, 
        root_encode=root_encode, 
        binned_targs=binned_targs,
        add_hs=add_hs
    )
    # Build out frag datasets
    train_dataset = dag_data.IntenDataset(
        train_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,        
        tree_processor=tree_processor,
    )
    val_dataset = dag_data.IntenDataset(
        val_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
    )

    test_dataset = dag_data.IntenDataset(
        test_df,
        data_dir=data_dir,
        magma_map=name_to_json,
        num_workers=num_workers,
        tree_processor=tree_processor,
    )

    # Update the config 
    config["model"]["params"]["node_feats"] = train_dataset.get_node_feats()

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=config["process_setting"]["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=config["process_setting"]["batch_size"],
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=config["process_setting"]["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=config["process_setting"]["batch_size"],
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=config["process_setting"]["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=config["process_setting"]["batch_size"],
    )

    return train_loader, val_loader, test_loader

def train(config):

    # Set a random seed 
    seed_everything(config["seed"])

    # Get the name of the dataset and the split 
    dataset = config["dataset"]["dataset_name"]
    split = config["dataset"]["split_name"].replace(".tsv", "")

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], "inten")
    expt_name = f"{dataset}_{split}"
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

    # Get the data loaders 
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Create model
    model = IntenGNN(**config["model"]["params"])

    # Get trainer and logger
    monitor = "val_loss"
    earlystop_callback = EarlyStopping(monitor=monitor, patience=20)
    checkpoint_callback = ModelCheckpoint(monitor=monitor,
                                          dirpath = results_dir,
                                          filename = '{epoch:02d}-{val_loss:.3f}',
                                          every_n_train_steps = config["trainer"]["log_every_n_steps"], 
                                          save_top_k = 2, mode = "min")

    trainer = pl.Trainer(**config["trainer"], 
                        logger = wandb_logger, 
                        callbacks=[checkpoint_callback, earlystop_callback])

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "inten_config.yaml", help = "Config file")
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--results_dir", type = str, default = "./results", help = "Results output directory")
    parser.add_argument("--debug", action = "store_true", default = False, help = "Set debug mode")
    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", default = True, help = "Enable wandb logging")
    parser.add_argument("--user", type = str, default = "serenakhoolm", help = "Set the user")

    args = parser.parse_args()

    # Set torch hub cache
    torch.hub.set_dir(args.torch_hub_cache)

    # # Read in the config
    config = read_config(os.path.join(args.config_dir, args.config_file))

    # Update the config 
    config["args"] = args.__dict__

    # Run the trainer now 
    train(config)
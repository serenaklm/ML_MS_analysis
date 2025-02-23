import os
import copy
import yaml
import argparse

import pandas as pd
from pathlib import Path

import torch 
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from utils import read_config
import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
from ms_pred.graff_ms import graff_ms_data

from model.graff_ms_model import GraffGNN

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

    subform_stem = config["dataset"]["form_dir_name"]
    subformula_folder = Path(data_dir) / "subformulae" / subform_stem
    form_map = {i.stem: Path(i) for i in subformula_folder.glob("*.json")}
    graph_featurizer = nn_utils.MolDGLGraph(pe_embed_k = config["process_setting"]["pe_embed_k"])

    # Update the config with the stats 
    config["model"]["params"]["atom_feats"] = graph_featurizer.atom_feats
    config["model"]["params"]["bond_feats"] = graph_featurizer.bond_feats
    config["model"]["params"]["num_atom_feats"] = graph_featurizer.num_atom_feats
    config["model"]["params"]["num_bond_feats"] = graph_featurizer.num_bond_feats

    num_bins = config["process_setting"]["num_bins"]
    upper_limit = config["process_setting"]["upper_limit"]
    num_workers = config["process_setting"].get("num_workers", 0)
    batch_size = config["process_setting"]["batch_size"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    train_dataset = graff_ms_data.BinnedDataset(
        train_df,
        form_map=form_map,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        graph_featurizer=graph_featurizer,
    )
    val_dataset = graff_ms_data.BinnedDataset(
        val_df,
        form_map=form_map,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        graph_featurizer=graph_featurizer,
    )
    test_dataset = graff_ms_data.BinnedDataset(
        test_df,
        form_map=form_map,
        data_dir=data_dir,
        num_bins=num_bins,
        # num_workers=num_workers,
        upper_limit=upper_limit,
        graph_featurizer=graph_featurizer,
    )

    # COnstruct the vocabulary
    # Losses will be negatives and others will be positives
    num_fixed_forms = config["process_setting"]["num_fixed_forms"]
    top_forms = train_dataset.get_top_forms()
    num_fixed_forms = min(num_fixed_forms, len(top_forms["forms"]))
    fixed_forms = top_forms["forms"][:num_fixed_forms]

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader, fixed_forms

def train(config):

    # Set a random seed 
    seed_everything(config["seed"])

    # Some sanity check 
    assert config["process_setting"]["num_bins"] == config["model"]["params"]["output_dim"]
    assert config["process_setting"]["upper_limit"] == config["model"]["params"]["upper_limit"]
    assert config["process_setting"]["num_fixed_forms"] == config["model"]["params"]["num_fixed_forms"]

    # Get the name of the dataset and the split 
    dataset = config["dataset"]["dataset_name"]
    split = config["dataset"]["split_name"].replace(".tsv", "")

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], "graff_ms")
    expt_name = f"{dataset}_{split}"
    # create_results_dir(results_dir)

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
    # write_config_local(config_o, os.path.join(results_dir, "run.yaml"))

    # Get the dataloaders
    train_loader, val_loader, test_loader, fixed_forms = get_dataloaders(config)

    # Get the model 
    model = GraffGNN(**config["model"]["params"])
    model.set_fixed_forms(fixed_forms)

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

    print() 

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

    # Update the config 
    config["args"] = args.__dict__

    # Run the trainer now 
    train(config)
import os
import copy
import yaml
import logging
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import read_config
from mist.data import datasets, splitter, featurizers

# Refine the mist model in our own directory 
from model import mist_model

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

def update_config(args, config):

    config["args"] = args.__dict__

    config["train_params"]["weight_decay"] = float(config["train_params"]["weight_decay"])
    config["model"]["params"]["fp_names"] = config["dataset"]["fp_names"] 
    config["model"]["params"]["magma_modulo"] = config["dataset"]["magma_modulo"]
    config["model"]["params"]["magma_aux_loss"] = config["dataset"]["magma_aux_loss"]

    config["model"]["params"]["learning_rate"] = config["train_params"]["learning_rate"] 
    config["model"]["params"]["weight_decay"] = config["train_params"]["weight_decay"]
    config["model"]["params"]["lr_decay_frac"] = config["train_params"]["lr_decay_frac"]
    config["model"]["params"]["scheduler"] = config["train_params"]["scheduler"]

    data_folder = config["dataset"]["data_folder"]
    dataset = config["dataset"]["dataset"]
    config["dataset"]["labels_file"] = os.path.join(data_folder, dataset, "labels.tsv")
    config["dataset"]["subform_folder"] = os.path.join(data_folder, dataset, "subformulae", "default_subformulae/")
    config["dataset"]["spec_folder"] = os.path.join(data_folder, dataset, "spec_folder")
    config["dataset"]["magma_folder"] = os.path.join(data_folder, dataset, "magma_outputs", "magma_tsv")
    config["dataset"]["split_file"] = os.path.join(data_folder, dataset, "splits", config["dataset"]["split_filename"])

    return config
    
def get_datamodule(config):

    # Split data
    my_splitter = splitter.get_splitter(**config["dataset"])

    # Get model class
    model_class = mist_model.MistNet
    config["model"]["name"] = model_class.__name__

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**config["dataset"])

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**config["dataset"])
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Redefine splitter s.t. this splits three times and remove subsetting
    split_name, (train, val, test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val", "test"], [train, val, test]):
        logging.info(f"Split: {split_name}, Len of {name}: {len(_data)}")
    
    train_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=train, featurizer=paired_featurizer, **config["train_settings"]
    )
    val_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=val, featurizer=paired_featurizer, **config["train_settings"]
    )
    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=test, featurizer=paired_featurizer, **config["train_settings"]
    )
    spec_dataloader_module = datasets.SpecDataModule(
        train_dataset, val_dataset, test_dataset, **config["train_settings"]
    ) # Note: this is already a pytorch lightning data module 

    return spec_dataloader_module

def train(config):

    # Set a random seed 
    seed_everything(config["seed"])

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], "mist")
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

    # Get the datamodule 
    datamodule = get_datamodule(config)

    # Create model
    model = mist_model.MistNet(**config["model"]["params"])
    # if config["model"].get("ckpt_file") is not None:
    #     model.load_from_ckpt(config["model"].get("ckpt_file"))

    # Get trainer and logger
    monitor = config["callbacks"]["val_monitor"]
    checkpoint_callback = ModelCheckpoint(monitor=monitor,
                                          dirpath = results_dir,
                                          filename = '{epoch:03d}-{val_bce_loss:.5f}', # Hack 
                                          every_n_train_steps = config["trainer"]["log_every_n_steps"], 
                                          save_top_k = 2, mode = "min")
    
    earlystop_callback = EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"])
    trainer = pl.Trainer(**config["trainer"], logger = wandb_logger, callbacks=[checkpoint_callback, earlystop_callback])

    # Start the training now
    trainer.fit(model, datamodule = datamodule)

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
    config = update_config(args, config)

    # Run the trainer now 
    train(config)
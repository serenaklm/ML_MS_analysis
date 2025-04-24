import os
import copy
import yaml
import random
import logging
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from mist.data import datasets, splitter, featurizers
from utils import read_config, consolidate_sampling_probability_IF, load_pickle, pickle_data

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

def sample_train(config):

    meta = ""
    if config["args"]["config_file"] == "wo_meta_config.yaml": meta = "wo_meta"
    elif config["args"]["config_file"] == "w_meta_config.yaml": meta = "w_meta"
    elif config["args"]["config_file"] == "sieved_config.yaml": meta = ""
    else: raise NotImplementedError() 

    dataset_name = config["dataset"]["dataset"]
    dataset_code = ""
    if dataset_name == "canopus": dataset_code = "C"
    if dataset_name == "massspecgym": dataset_code = "MSG"
    if dataset_name == "nist2023": dataset_code = "NIST2023"

    split_name = config["dataset"]["split_filename"].replace(".tsv", "")

    if meta == "w_meta":
        folder = os.path.join("./best_models/", dataset_name, f"{dataset_code}_MIST_meta_4096_{split_name}")
    elif meta == "wo_meta":
        folder = os.path.join("./best_models/", dataset_name, f"{dataset_code}_MIST_4096_{split_name}")

    # Get the sampling prob 
    train_sample_prob_path = os.path.join(folder, "influence_score.pkl")
    if not os.path.exists(train_sample_prob_path):
        
        print("Need to generate the sampling probability now")
        score_dict = consolidate_sampling_probability_IF(folder)
        pickle_data(score_dict, train_sample_prob_path)

    # Check if we are removing random samples or based on IF 
    random_check = config["args"]["random"] 

    # Sample train records to add to the dataset
    ratio = config["args"]["sampling_ratio"]
    score_dict = load_pickle(train_sample_prob_path)
    n_train_to_select = int(ratio * len(score_dict))

    if random_check: 
        selected_ids = list(score_dict.keys())
        selected_ids = random.sample(selected_ids, n_train_to_select)

    else:

        score_dict = sorted(score_dict.items(), key=lambda item: item[1], reverse = True) # descending order   
        selected_ids = [p[0] for p in score_dict[:n_train_to_select]] # Keep the top k 
    
    selected_ids = [k.replace(".ms", "") for k in selected_ids]

    return selected_ids

def get_datamodule(config):

    # Split data
    my_splitter = splitter.get_splitter(**config["dataset"])

    # Check if we are going to sample the dataset 
    sampling_ratio = config["args"]["sampling_ratio"]

    if sampling_ratio != 0.0: 
        print("Sampling training data now")
        if config["args"]["random"]:
            print("Random sampling now")
        assert sampling_ratio > 0.0 and sampling_ratio < 1.0 

        # Get the samples to use for training
        selected_ids = sample_train(config)

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

    # Sample it now
    if sampling_ratio != 0.0:
        train = [p for p in train if p[0].spectra_name in selected_ids]

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

def get_exp_name(config):

    dataset_code = ""

    if "canopus" in config["dataset"]["dataset"]: dataset_code = "C"
    elif "massspecgym" in config["dataset"]["dataset"]: dataset_code = "MSG"
    elif "nist2023" in config["dataset"]["dataset"]: dataset_code = "NIST2023"
    else: raise Exception("Dataset not recognized - ", config["dataset"]["dataset"])

    model_code = "MIST"
    split_code = config["dataset"]["split_file"].split("/")[-1].replace(".tsv", "")

    if "w_meta" in config["args"]["config_file"]:
        name = f"{dataset_code}_{model_code}_meta_4096_{split_code}"
    elif "wo_meta" in config["args"]["config_file"]: 
        name = f"{dataset_code}_{model_code}_4096_{split_code}"

    elif "sieved" in config["args"]["config_file"]: 
        name = f"{dataset_code}_{model_code}_sieved_4096_{split_code}"
    else:
        raise NotImplementedError() 
    
    ratio = config["args"]["sampling_ratio"]
    if ratio != 0.0:
        ratio = int(ratio * 100)
        name = f"{name}_{ratio}"
    
    random_check = config["args"]["random"] 
    if random_check: 
        name = f"{name}_random"

    return name 

def train(config):

    # Set a random seed 
    seed_everything(config["seed"])

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], "mist")
    expt_name = get_exp_name(config)

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
    parser.add_argument("--config_file", type = str, default = "sieved_config.yaml", help = "Config file")
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--results_dir", type = str, default = "./results", help = "Results output directory")
    parser.add_argument("--debug", action = "store_true", default = False, help = "Set debug mode")
    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", default = True, help = "Enable wandb logging")
    parser.add_argument("--sampling_ratio", type = float, default = 0.0, help = "Remove top k most harmful samples from the training set")
    parser.add_argument("--random", action = "store_true", default = False, help = "Randomly selecting samples to remove from the training set")
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

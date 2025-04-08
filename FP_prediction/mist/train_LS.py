import os
import yaml
import copy
import random
import argparse
from typing import List
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from modules import Data
from utils import read_config

from model import mist_model
from mist.data import datasets, featurizers
from learning_to_split import set_seed, get_optim, split_data, train_splitter

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
    config["dataset"]["FP_key"] = "mols"

    # Add the device 
    device_id = config["device"]
    config["device"] = torch.device(f"cuda:{device_id}")

    return config
    
@rank_zero_only
def write_config_local(config, config_out_path):
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)

@rank_zero_only
def create_results_dir(results_dir):
    if not os.path.exists(results_dir): os.makedirs(results_dir)

def get_data_modules(spectra_mol_pairs, train_indices, test_indices, paired_featurizer):

    random.shuffle(train_indices)
    n_train = int(0.8 * len(train_indices))

    train_dataset = datasets.SpectraMolDataset(
        spectra_mol_list = [spectra_mol_pairs[i] for i in train_indices[:n_train]], featurizer=paired_featurizer, **config["train_settings"])

    val_dataset = datasets.SpectraMolDataset(
        spectra_mol_list = [spectra_mol_pairs[i] for i in train_indices[n_train:]], featurizer=paired_featurizer, **config["train_settings"])

    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list = [spectra_mol_pairs[i] for i in test_indices], featurizer=paired_featurizer, **config["train_settings"])

    datamodule = datasets.SpecDataModule(train_dataset, val_dataset, test_dataset, **config["train_settings"])

    return datamodule, (val_dataset, test_dataset)

def learning_to_split(config: dict, 
                      verbose: bool = True):

    # Get the dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**config["dataset"])
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))
    paired_featurizer = featurizers.get_paired_featurizer(**config["dataset"])
    dataset = Data(data = spectra_mol_pairs, train_mode = True, featurizer = paired_featurizer)

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"], "mist")
    expt_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

    # Initialize the logger
    wandb_logger = None
    if not config["args"]["debug"] and config["args"]["wandb"]:
        wandb_logger = WandbLogger(project = config["project"],
                                    config = config,
                                    group = config["args"]["config_file"].replace(".yaml", ""),
                                    entity = config["args"]["user"],
                                    name = expt_name,
                                    log_model = False)
    config["run"] = wandb_logger 
    
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
    config_splitter["run"] = wandb_logger
    config_splitter["dataset"]["fp_names"] = ["splitter"]
    config_splitter["model"]["params"]["fp_names"] = config_splitter["dataset"]["fp_names"]
    splitter = mist_model.MistNet(**config_splitter["model"]["params"])
    opt = get_optim(splitter, config)

    # Start training here
    for outer_loop in range(config["train_params"]["n_outer_loops"]):

        predictor = mist_model.MistNet(**config["model"]["params"])

        # Split the data 
        random_split = True if outer_loop == 0 else False
        split_stats, train_indices, test_indices = split_data(dataset, splitter, 
                                                              config["splitter"]["train_ratio"], 
                                                              config["train_settings"]["batch_size"], 
                                                              config["train_settings"]["num_workers"], 
                                                              random_split) 

        # Get the data modules now 
        datamodule, (val_dataset, test_dataset) = get_data_modules(spectra_mol_pairs, train_indices, test_indices, paired_featurizer)

        # Get trainer and logger
        monitor = config["callbacks"]["val_monitor"]
        earlystop_callback = EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"])
        trainer = pl.Trainer(**config["trainer"], logger = None, callbacks=[earlystop_callback])

        # Start the training now
        # trainer.fit(predictor, datamodule = datamodule)

        # # Get the gap 
        # val_loss = trainer.validate(model = predictor, dataloaders = datamodule)[0]["val_loss"]
        # test_loss = trainer.test(model = predictor, dataloaders = datamodule)[0]["test_loss"]
        # gap = test_loss - val_loss

        # if gap > best_gap:
            
        #     best_gap, num_no_improvements = gap, 0

            # best_split = {"train_indices":  train_indices,
            #               "test_indices":   test_indices,
            #               "val_loss":      val_loss,
            #               "test_loss":     test_loss,
            #               "split_stats":    split_stats,
            #               "outer_loop":     outer_loop,
            #               "best_gap":       best_gap}
            
            # save(splitter, predictor, best_split, config["output_dir"])

        # else: num_no_improvements += 1
        # if num_no_improvements == config["splitter"]["patience"]: break

        # Train the splitter
        train_splitter(splitter, predictor, dataset, test_indices, opt,
                       config["train_settings"]["batch_size"], config["splitter"]["num_batches"], config["train_settings"]["num_workers"],
                       config, verbose = verbose)
        
    # Done! Print the best split.
    if verbose:
        print("Finished!\nBest split:")
        print_split_status(best_split["outer_loop"], best_split["split_stats"],
                           best_split["val_score"], best_split["test_score"])

    a = z 

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

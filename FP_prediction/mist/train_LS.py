import os
import yaml
import copy
import random
import argparse
import pandas as pd
from functools import partial
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from modules import Data
from utils import read_config, pickle_data, write_json

from model import mist_model
from mist.data import datasets, featurizers
from mist.data.datasets import _collate_pairs

from learning_to_split import split_data

os.environ["WANDB_API_KEY"] = "d72d59862e6a3d35823879bd4078f5199bc26639"

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

    return datamodule

def get_data_modules_splitter(spectra_mol_pairs, test_indices, paired_featurizer):

    dataset = datasets.SpectraMolDataset(
        spectra_mol_list = [spectra_mol_pairs[i] for i in test_indices], featurizer=paired_featurizer, **config["train_settings"])

    datamodule = datasets.SpecDataModule(dataset, dataset, dataset, **config["train_settings"]) # no difference in the set, model just needs to learn

    return datamodule

def learning_to_split(config: dict, 
                      verbose: bool = True):

    # Set a random seed 
    seed_everything(config["seed"])

    # Get the dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**config["dataset"])
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Get the permitted Ids 
    id_list = pd.read_csv(config["dataset"]["split_file"], sep = "\t").loc[:, "name"].values.tolist()
    id_list = [str(i) for i in id_list]

    # filter out IDs that are not found in the list 
    spectra_mol_pairs = [p for p in spectra_mol_pairs if str(p[0].spectra_name) in id_list]
 
    paired_featurizer = featurizers.get_paired_featurizer(**config["dataset"])

    mol_collate_fn = paired_featurizer.get_mol_collate()
    spec_collate_fn = paired_featurizer.get_spec_collate()
    
    collate_pairs = partial(_collate_pairs,
                            mol_collate_fn=mol_collate_fn,
                            spec_collate_fn=spec_collate_fn)

    dataset = Data(data = spectra_mol_pairs, train_mode = True, featurizer = paired_featurizer)
    dataloader = DataLoader(dataset, batch_size = config["train_settings"]["batch_size"], 
                                                  shuffle = False,
                                                  num_workers=config["train_settings"]["num_workers"],
                                                  collate_fn = collate_pairs)

    # Update the results directory 
    results_dir = os.path.join(config["args"]["results_dir"])
    dataset_name = config["dataset"]["dataset"]
    expt_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    expt_name = f"MIST_{dataset_name}_{expt_name}"
    results_dir = os.path.join(results_dir, expt_name)
    create_results_dir(results_dir)

    # Write the data ids for future analysis
    data_ids = [p[0].spectra_name for p in spectra_mol_pairs]
    pickle_data(data_ids, os.path.join(results_dir, "data_ids.pkl"))
    
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
    # Remove params that we need for FP prediction but not for the splitter 
    splitter_model_params = copy.deepcopy(config["model"]["params"])
    splitter_model_params["tar_ratio"] = train_ratio
    splitter_model_params["w_gap"] = config["splitter"]["w_gap"]
    splitter_model_params["w_ratio"] = config["splitter"]["w_ratio"]
    splitter_model_params["w_balance"] = config["splitter"]["w_balance"]

    del splitter_model_params["magma_loss_lambda"]
    del splitter_model_params["iterative_preds"]
    del splitter_model_params["iterative_loss_weight"]
    del splitter_model_params["shuffle_train"]
    del splitter_model_params["loss_fn"]
    del splitter_model_params["pos_weight"]
    del splitter_model_params["refine_layers"]
    del splitter_model_params["fp_names"]
    del splitter_model_params["magma_modulo"]
    del splitter_model_params["magma_aux_loss"]

    splitter = mist_model.MistNetSplitter(**splitter_model_params)

    # Get the wandb logger 
    wandb_logger = WandbLogger(project = config["project"],
                               config = config,
                               group = config["args"]["config_file"].replace(".yaml", ""),
                               entity = config["args"]["user"],
                               name = expt_name,
                               log_model = False)

    # Start training here
    for outer_loop in range(config["train_params"]["n_outer_loops"]):

        predictor = mist_model.MistNet(**config["model"]["params"])

        # Split the data 
        random_split = True if outer_loop == 0 else False
        split_stats, train_indices, test_indices = split_data(dataloader, splitter, 
                                                              config["splitter"]["train_ratio"], 
                                                              random_split) 

        # Get the data modules now 
        datamodule = get_data_modules(spectra_mol_pairs, train_indices, test_indices, paired_featurizer)

        # Get trainer and logger
        monitor = config["callbacks"]["val_monitor"]
        checkpoint_callback = ModelCheckpoint(monitor=monitor,
                                              dirpath = results_dir,
                                              filename = 'best_predictor', # Hack 
                                              save_top_k = 1, mode = "min")
        earlystop_callback = EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"])
        trainer = pl.Trainer(**config["trainer"], logger = wandb_logger, callbacks=[checkpoint_callback, earlystop_callback])

        # Start the training now
        trainer.fit(predictor, datamodule = datamodule)

        # Get the gap 
        val_loss = trainer.validate(model = predictor, dataloaders = datamodule)[0]["val_loss"]
        test_loss = trainer.test(model = predictor, dataloaders = datamodule)[0]["test_loss"]
        gap = test_loss - val_loss

        # Add to wandb 
        wandb_logger.log_metrics({"predictor/val_loss_epoch": val_loss}, step = outer_loop)
        wandb_logger.log_metrics({"predictor/test_loss_epoch": test_loss}, step = outer_loop)
        wandb_logger.log_metrics({"predictor/gap_epoch": gap}, step = outer_loop)

        if gap > best_gap:
            
            best_gap, num_no_improvements = gap, 0

            best_split = {"train_indices":  train_indices,
                            "test_indices":    test_indices,
                            "val_loss":        val_loss,
                            "test_loss":       test_loss,
                            "split_stats":     split_stats,
                            "outer_loop":      outer_loop,
                            "best_gap":        best_gap}
                
            pickle_data(best_split, os.path.join(results_dir, "best_split.pkl"))

            # Write down some stats of the current split too 
            split_stats["best_gap"] = best_gap
            write_json(split_stats, os.path.join(results_dir, "splits_stats.json"))

            # Save the state dict of the splitter 
            torch.save(splitter.state_dict(), os.path.join(results_dir, "best_splitter.pth"))
 
        else: num_no_improvements += 1
        if num_no_improvements == config["splitter"]["patience"]: break

        # Add predictor to the splitter 
        splitter.add_predictor(predictor)

        # Get the splitter datamodule 
        splitter_datamodule = get_data_modules_splitter(spectra_mol_pairs, test_indices, paired_featurizer)

        # Train the splitter
        splitter_monitor = config["splitter"]["monitor"]
        splitter_checkpoint_callback = ModelCheckpoint(monitor = splitter_monitor,
                                                       dirpath = results_dir,
                                                       filename = 'latest_splitter',
                                                       every_n_epochs = config["splitter"]["every_n_epochs"])
        splitter_earlystop_callback = EarlyStopping(monitor=splitter_monitor, patience=config["splitter"]["patience"])
        splitter_trainer = pl.Trainer(**config["splitter_trainer"], logger = wandb_logger, callbacks=[splitter_checkpoint_callback, splitter_earlystop_callback])
        splitter_trainer.fit(splitter, datamodule = splitter_datamodule)

    # Done! Print the best split.
    if verbose:
        print("Finished!\nBest split:")
        print(best_split["outer_loop"], best_split["split_stats"],
                        best_split["val_loss"], best_split["test_loss"])

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

    # Run learning to split now 
    learning_to_split(config)

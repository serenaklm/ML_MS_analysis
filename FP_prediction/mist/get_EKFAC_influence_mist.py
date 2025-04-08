import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.analyzer import Analyzer, prepare_model

from modules import * 
from mist.data import datasets, splitter, featurizers
from utils import read_config, load_pickle, pickle_data

from model.mist_model import MistNet

class TaskMIST(Task):

    def compute_train_loss(self,
        batch: Any,
        model: nn.Module,
        sample: bool = False) -> torch.Tensor:

        # Sum pool over channels for simplicity
        pred_fp, aux_outputs_spec = model.encode_spectra(batch)

        # Mol fp's
        target_fp_all, aux_outputs_mol = model.encode_mol(batch)
        target_fp_all = target_fp_all.float()

        # Get the proper matched indices
        mol_inds = batch["mol_indices"]
        norm_inds = mol_inds[batch["matched"]]
        target_fp = target_fp_all[norm_inds]

        # Compute loss and return loss
        fingerprints = batch.get("fingerprints")
        fingerprints[fingerprints == -1] = 0
        
        ret_dict = model.compute_loss(
            pred_fp,
            target_fp,
            aux_outputs_mol=aux_outputs_mol,
            aux_outputs_spec=aux_outputs_spec,
            fingerprints=fingerprints,
            fingerprint_mask=batch.get("fingerprint_mask"),
            train_step=True)

        return ret_dict["loss"]

    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module) -> torch.Tensor:

        # Sum pool over channels for simplicity
        pred_fp, aux_outputs_spec = model.encode_spectra(batch)

        # Mol fp's
        target_fp_all, aux_outputs_mol = model.encode_mol(batch)
        target_fp_all = target_fp_all.float()

        # Get the proper matched indices
        mol_inds = batch["mol_indices"]
        norm_inds = mol_inds[batch["matched"]]
        target_fp = target_fp_all[norm_inds]

        # Compute loss and return loss
        fingerprints = batch.get("fingerprints")
        fingerprints[fingerprints == -1] = 0

        ret_dict = model.compute_loss(
            pred_fp,
            target_fp,
            aux_outputs_mol=aux_outputs_mol,
            aux_outputs_spec=aux_outputs_spec,
            fingerprints=fingerprints,
            fingerprint_mask=batch.get("fingerprint_mask"),
            train_step=True)

        return ret_dict["loss"]

    def get_influence_tracked_modules(self) -> Optional[List[str]]:

        total_modules = []

        # Input encoder 
        total_modules.append(f"spectra_encoder.0.intermediate_layer.input_layer")
        total_modules.append(f"spectra_encoder.0.intermediate_layer.layers.0")

        # Pairwise featurizer
        total_modules.append(f"spectra_encoder.0.pairwise_featurizer.input_layer")
        total_modules.append(f"spectra_encoder.0.pairwise_featurizer.layers.0")

        # The predictors 
        total_modules.append(f"spectra_encoder.1.0")
        total_modules.append(f"spectra_encoder.2.initial_predict.0")

        for i in range(4):
            total_modules.append(f"spectra_encoder.2.gate_bricks.{i}.0")
            total_modules.append(f"spectra_encoder.2.predict_bricks.{i}.0")

        return total_modules

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:

        return batch["fingerprint_mask"]

def update_params(params):

    params["train_params"]["weight_decay"] = float(params["train_params"]["weight_decay"])
    params["model"]["params"]["fp_names"] = params["dataset"]["fp_names"] 
    params["model"]["params"]["magma_modulo"] = params["dataset"]["magma_modulo"]
    params["model"]["params"]["magma_aux_loss"] = params["dataset"]["magma_aux_loss"]

    params["model"]["params"]["learning_rate"] = params["train_params"]["learning_rate"] 
    params["model"]["params"]["weight_decay"] = params["train_params"]["weight_decay"]
    params["model"]["params"]["lr_decay_frac"] = params["train_params"]["lr_decay_frac"]
    params["model"]["params"]["scheduler"] = params["train_params"]["scheduler"]

    data_folder = params["dataset"]["data_folder"]
    dataset = params["dataset"]["dataset"]
    params["dataset"]["labels_file"] = os.path.join(data_folder, dataset, "labels.tsv")
    params["dataset"]["subform_folder"] = os.path.join(data_folder, dataset, "subformulae", "default_subformulae/")
    params["dataset"]["spec_folder"] = os.path.join(data_folder, dataset, "spec_folder")
    params["dataset"]["magma_folder"] = os.path.join(data_folder, dataset, "magma_outputs", "magma_tsv")
    params["dataset"]["split_file"] = os.path.join(data_folder, dataset, "splits", params["dataset"]["split_filename"])

    return params

def get_checkpoint_path(folder):

    checkpoints = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    best_checkpoint, lowest_loss = "", 1e4

    for c in checkpoints:

        loss = float(c.replace("-v1", "").replace(".ckpt", "").split("=")[-1]) # hack 
        if loss < lowest_loss:
            lowest_loss = loss 
            best_checkpoint = c 
    
    return os.path.join(folder, best_checkpoint)

def get_datasets(folder, params, top_k):

    # Split data
    my_splitter = splitter.get_splitter(**params["dataset"])

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**params["dataset"])

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**params["dataset"])
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Get the train dataset 
    _, (train, _, test) = my_splitter.get_splits(spectra_mol_pairs)

    # Get the test id_ to analyze
    test_results_path = os.path.join(folder, "test_results.pkl")
    test_results = load_pickle(test_results_path)
    test_results = sorted(test_results.items(), key = lambda item: item[1]["loss"], reverse = True)
    
    test_ids_to_analyze = [str(r[0]).replace("tensor(", "").replace(")", "") for r in test_results[:top_k]] # Hack
    test = [r for r in test if r[0].spectra_name in test_ids_to_analyze]

    # Get the train ids 
    train_files = [r[0].spectra_file for r in train]
    test_files = [r[0].spectra_file for r in test]

    # Build dataset
    train_dataset = Data(data = train, 
                        train_mode = True,
                        featurizer = paired_featurizer)
    
    test_dataset = Data(data = test, 
                        train_mode = True,
                        featurizer = paired_featurizer)

    print(f"Analyzing : {len(test)} test samples")

    return train_dataset, test_dataset, train_files, test_files

def get_modules(model_cache_folder, params, top_k):

    # Get the model 
    model = MistNet.load_from_checkpoint(get_checkpoint_path(model_cache_folder)).train()

    # Get the task 
    task = TaskMIST()

    # Get the datasets
    train_data, test_data, train_ids, test_ids = get_datasets(model_cache_folder, params, top_k)

    return model, task, train_data, test_data, train_ids, test_ids

def get_influence_scores(folder, output_path, top_k):

    # Get the parameters 
    params = read_config(folder / "run.yaml")
    params = update_params(params)

    # Get the model 
    model, task, train_data, test_data, train_ids, test_ids = get_modules(folder, params, top_k)

    # Get the modules to get the influence scores 
    model = prepare_model(model = model, task = task) 
    analyzer = Analyzer(analysis_name = f"{folder.stem}", model = model, task = task)

    # [Optional] Set up the parameters for the DataLoader.
    dataloader_kwargs = DataLoaderKwargs(num_workers=4, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute all factors
    analyzer.fit_all_factors(factors_name="EKFAC", dataset=train_data)

    # Get the scores
    analyzer.compute_pairwise_scores(
        scores_name="scores",
        factors_name="EKFAC",
        query_dataset=test_data,
        train_dataset=train_data,
        per_device_query_batch_size=16,
        per_device_train_batch_size=128,
    )

    # Load the scores 
    scores = analyzer.load_pairwise_scores(scores_name = "scores")

    # Save the scores
    pickle_data(scores, output_path)

    # Save the ids
    pickle_data(train_ids, folder / "train_ids.pkl")
    pickle_data(test_ids, folder / "test_ids.pkl")

if __name__ == "__main__":

    top_k = 1000

    # Manually add all folders to be processed into a list (hack)
    folder = "./models_cached/"
    all_folders = []
    
    for FP in os.listdir(folder):
        FP_folder = os.path.join(folder, FP)
        for dataset in os.listdir(FP_folder):
            dataset_folder = os.path.join(FP_folder, dataset)
            for checkpoint in os.listdir(dataset_folder):
                all_folders.append(os.path.join(dataset_folder, checkpoint))

    # Iterate through all folders to get influence scores for the models
    for f in all_folders:

        f = Path(f)
        output_path = f / "EK-FAC_scores.pkl"
        if os.path.exists(output_path): print(f"{output_path} already exists. Continue.")

        print(f"Getting the influence scores for: {f}")
        get_influence_scores(f, output_path, top_k)

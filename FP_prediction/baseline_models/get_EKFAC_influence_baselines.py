import os 
import random 
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from kronfluence.task import Task
from kronfluence.arguments import ScoreArguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.analyzer import Analyzer, prepare_model

from modules import * 
from dataloader import MSDataset, Data
from utils import read_config, load_pickle, pickle_data, replace_MS_encoder_layers

class TaskBinnedMS(Task):

    def __init__(self, include_adduct, include_CE, include_instrument):
        super().__init__()
        self.include_adduct = include_adduct
        self.include_CE = include_CE
        self.include_instrument = include_instrument

    def compute_train_loss(self,
        batch: Any,
        model: nn.Module,
        sample: bool = False) -> torch.Tensor:

        # Unpack the batch 
        FP = batch["FP"]
        binned_ms = batch["binned_MS"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        FP_pred, binned_ms_pred = model(binned_ms, adduct, CE, instrument)

        # Get the loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + 1e-6 * reconstruction_loss

    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module) -> torch.Tensor:

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass 
        FP_pred, binned_ms_pred = model(batch["binned_MS"], adduct, CE, instrument)

        # Get the loss 
        reconstruction_loss = F.mse_loss(binned_ms_pred, batch["binned_MS"])
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, batch["FP"])

        return FP_loss + 1e-6 * reconstruction_loss

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        return None

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        return None

class TaskMS(Task):

    def __init__(self, include_adduct, include_CE, include_instrument, n_layers):
        super().__init__()
        self.include_adduct = include_adduct
        self.include_CE = include_CE
        self.include_instrument = include_instrument
        self.n_layers = n_layers 

    def compute_train_loss(self,
        batch: Any,
        model: nn.Module,
        sample: bool = False) -> torch.Tensor:

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        FP_pred, binned_ms_pred = model(mz, intensities, mask, binned_ms, adduct, CE, instrument)
        
        # Get loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + 1e-6 * reconstruction_loss

    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module) -> torch.Tensor:

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        FP_pred, binned_ms_pred = model(mz, intensities, mask, binned_ms, adduct, CE, instrument)
        
        # Get loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + 1e-6 * reconstruction_loss

    def get_influence_tracked_modules(self) -> Optional[List[str]]:

        total_modules = []

        # For the mz encoder 
        for i in [0,2]:
            total_modules.append(f"mz_encoder.MLP.{i}")


        # For the intensity encoder
        for i in [0,2]:
            total_modules.append(f"intensity_encoder.MLP.{i}")

        # For the peak encoder
            total_modules.append(f"peaks_encoder.{i}")
        
        # For the main chunk 
        for i in range(self.n_layers):
            total_modules.append(f"MS_encoder.layers.{i}.linear1")
            total_modules.append(f"MS_encoder.layers.{i}.linear2")

        # For the binned MS encoder
        for i in [0,3,6,9]:
            total_modules.append(f"binned_ms_encoder.{i}")

        # For the prediction layer 
        for i in [0,2]:
            total_modules.append(f"pred_layer.{i}")

        return total_modules

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:

        return batch["mask"]

class TaskFormula(Task):

    def __init__(self, include_adduct, include_CE, include_instrument, n_layers):

        super().__init__()
        self.include_adduct = include_adduct
        self.include_CE = include_CE
        self.include_instrument = include_instrument
        self.n_layers = n_layers 

    def compute_train_loss(self,
        batch: Any,
        model: nn.Module,
        sample: bool = False) -> torch.Tensor:

        # Unpack the batch 
        intensities, formula, mask = batch["intensities"], batch["formula"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        FP_pred, binned_ms_pred = model(intensities, formula, mask, binned_ms, adduct, CE, instrument)

        # Get loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + 1e-6 * reconstruction_loss

    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module) -> torch.Tensor:

        # Unpack the batch 
        intensities, formula, mask = batch["intensities"], batch["formula"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        FP_pred, binned_ms_pred = model(intensities, formula, mask, binned_ms, adduct, CE, instrument)

        # Get loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + 1e-6 * reconstruction_loss

    def get_influence_tracked_modules(self) -> Optional[List[str]]:

        total_modules = []

        # For the formula encoder 
        for i in [0,2]:
            total_modules.append(f"formula_encoder.{i}")

        # For the intensity encoder
        for i in [0,2]:
            total_modules.append(f"intensity_encoder.MLP.{i}")

        # For the peak encoder
            total_modules.append(f"peaks_encoder.{i}")

        # For the main chunk 
        for i in range(self.n_layers):
            total_modules.append(f"MS_encoder.layers.{i}.linear1")
            total_modules.append(f"MS_encoder.layers.{i}.linear2")

        # For the binned MS encoder
        for i in [0,3,6,9]:
            total_modules.append(f"binned_ms_encoder.{i}")

        # For the prediction layer 
        for i in [0,2]:
            total_modules.append(f"pred_layer.{i}")

        return total_modules
    
    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:

        return batch["mask"]

def get_checkpoint_path(folder):

    checkpoints = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    best_checkpoint, lowest_loss = "", 1e4

    for c in checkpoints:

        loss = float(c.replace(".ckpt", "").split("=")[-1]) # hack 
        if loss < lowest_loss:
            lowest_loss = loss 
            best_checkpoint = c 
    
    return os.path.join(folder, best_checkpoint)

def get_datasets(folder, params, top_k):

    params["data"]["get_CF"], params["data"]["get_frags"] = False, False
    model_name = params["model"]["name"]

    if model_name == "formula_encoder": params["data"]["get_CF"] = True 
    if model_name == "frag_encoder": params["data"]["get_frags"] = True 

    dataset = MSDataset(**params["data"])
    train_files_to_analyze = dataset.train_data
    train_data = Data(train_files_to_analyze, dataset.process)

    # Get the test id_ to analyze
    test_results_path = os.path.join(folder, "test_results.pkl")
    test_results = load_pickle(test_results_path)
    test_results = sorted(test_results.items(), key = lambda item: item[1]["loss"], reverse = True)

    # Look at 3 different types of test - the good test, the bad test and anything in the middle 
    test_ids_to_analyze_bad_mistakes = [str(r[0]).replace("tensor(", "").replace(")", "") for r in test_results[:top_k]] # Hack
    test_ids_to_analyze_good_predictions = [str(r[0]).replace("tensor(", "").replace(")", "") for r in test_results[-top_k:]] # Hack
    test_ids_to_analyze_random = [str(r[0]).replace("tensor(", "").replace(")", "") for r in random.sample(test_results[top_k: len(test_results) - top_k], min(top_k, len(test_results) - 2 * top_k))] # Hack
    test_ids_to_analyze = test_ids_to_analyze_bad_mistakes + test_ids_to_analyze_good_predictions + test_ids_to_analyze_random

    test_files_to_analyze = [f for f in dataset.test_data if Path(f).stem in test_ids_to_analyze]
    test_data = Data(test_files_to_analyze, dataset.process)
    print(f"Analyzing : {len(test_data)} test samples")
    
    return train_data, test_data, train_files_to_analyze, test_files_to_analyze

def get_modules(model_cache_folder, params, top_k, include_adduct, include_CE, include_instrument):

    model_path = get_checkpoint_path(model_cache_folder)

    model_name = params["model"]["name"] 
    if model_name == "binned_MS_encoder":
        model = MSBinnedModel.load_from_checkpoint(model_path).train()
        task = TaskBinnedMS(include_adduct, include_CE, include_instrument)

    elif model_name == "MS_encoder":
        model = MSTransformerEncoder.load_from_checkpoint(model_path).train()
        task = TaskMS(include_adduct, include_CE, include_instrument, params["model"]["MS_encoder"]["n_layers"])

    elif model_name == "formula_encoder":
        model = FormulaTransformerEncoder.load_from_checkpoint(model_path).train()
        task = TaskFormula(include_adduct, include_CE, include_instrument, params["model"]["MS_encoder"]["n_layers"])

    else:
        raise NotImplementedError()
    
    train_data, test_data, train_ids, test_ids = get_datasets(model_cache_folder, params, top_k)

    return model, task, train_data, test_data, train_ids, test_ids

def get_influence_scores(folder, output_path, self_output_path, top_k):

    # Get the parameters 
    params = read_config(folder / "run.yaml")

    # Update the params
    params["data"]["dir"] = os.path.join(params["data"]["data_folder"], params["data"]["dataset"], "frags_preds")
    params["data"]["split_file"] = os.path.join(params["data"]["splits_folder"], params["data"]["dataset"], "splits", params["data"]["split_file"])
    params["data"]["adduct_file"] = os.path.join(params["data"]["data_folder"], params["data"]["dataset"], "all_adducts.pkl")
    params["data"]["instrument_file"] = os.path.join(params["data"]["data_folder"], params["data"]["dataset"], "all_instruments.pkl")
    del params["data"]["dataset"]
    del params["data"]["data_folder"]
    del params["data"]["splits_folder"]

    # Set some params 
    feats_params = params["model"]["feats_params"]
    include_adduct, include_CE, include_instrument = feats_params["include_adduct"], feats_params["include_CE"], feats_params["include_instrument"]

    # Get the model 
    model, task, train_data, test_data, train_ids, test_ids = get_modules(folder, params, top_k, 
                                                                          include_adduct, include_CE, include_instrument)

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
        per_device_query_batch_size = 64,
        per_device_train_batch_size = 64)
    
    # Get the pairwise scores too
    analyzer.compute_self_scores(
        scores_name = "self_scores",
        factors_name = "EKFAC",
        train_dataset = train_data,
        per_device_train_batch_size = 64
    )

    # Load the scores 
    scores = analyzer.load_pairwise_scores(scores_name = "scores")
    self_scores = analyzer.load_self_scores(scores_name = "self_scores")

    # Save the scores
    pickle_data(scores, output_path)
    pickle_data(self_scores, self_output_path)

    # Save the ids
    pickle_data(train_ids, folder / "train_ids.pkl")
    pickle_data(test_ids, folder / "test_ids.pkl")

if __name__ == "__main__":

    top_k = 1000

    # Manually add all folders to be processed into a list (hack)
    folder = "./best_models/"
    all_folders = []
    
    for dataset in os.listdir(folder):
        if dataset != "massspecgym_sieved": continue
        dataset_folder = os.path.join(folder, dataset)
        for checkpoint in os.listdir(dataset_folder):
            all_folders.append(os.path.join(dataset_folder, checkpoint))

    # Iterate through all folders to get influence scores for the models
    for f in all_folders:

        f = Path(f)
        output_path = f / "EK-FAC_scores.pkl"
        self_output_path = f / "EK-FAC_self_scores.pkl"
        if os.path.exists(output_path) and os.path.exists(self_output_path): 
            print(f"{output_path} already exists. Continue.")
            continue

        print(f"Getting the influence scores for: {f}")
        get_influence_scores(f, output_path, self_output_path, top_k)
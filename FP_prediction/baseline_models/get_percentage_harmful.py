import os 
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from utils import read_config, load_pickle, pickle_data

from modules import * 
from dataloader import MSDataset

from torch_influence import BaseObjective, AutogradInfluenceModule, LiSSAInfluenceModule, CGInfluenceModule

class ObjectiveBinnedMS(BaseObjective):

    def train_outputs(self, model, batch):

        # Unpack the batch 
        binned_ms = batch["binned_MS"]

        # Forward pass
        FP_pred, binned_ms_pred = model(binned_ms)

        return FP_pred, binned_ms_pred

    def train_loss_on_outputs(self, outputs, batch):
        
        FP_pred, binned_ms_pred = outputs
        FP = batch["FP"]
        binned_ms = batch["binned_MS"]
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + reconstruction_loss

    def train_regularization(self, params):
        return 0.01 * torch.square(params.norm())

    def test_loss(self, model, params, batch):

        FP_pred, binned_ms_pred = model(batch["binned_MS"])
        reconstruction_loss = F.mse_loss(binned_ms_pred, batch["binned_MS"])
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, batch["FP"])

        return FP_loss + reconstruction_loss

class ObjectiveMS(BaseObjective):

    def train_outputs(self, model, batch):

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]

        # Forward pass
        FP_pred, binned_ms_pred = model(mz, intensities, mask, binned_ms)

        return FP_pred, binned_ms_pred

    def train_loss_on_outputs(self, outputs, batch):
        
        FP_pred, binned_ms_pred = outputs
        FP = batch["FP"]
        binned_ms = batch["binned_MS"]
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + reconstruction_loss

    def train_regularization(self, params):
        return 0.01 * torch.square(params.norm())

    def test_loss(self, model, params, batch):

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        # Forward pass
        FP_pred, binned_ms_pred = model(mz, intensities, mask, binned_ms)

        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + reconstruction_loss

class ObjectiveFormula(BaseObjective):

    def train_outputs(self, model, batch):

        # Unpack the batch 
        intensities, formula, mask = batch["intensities"], batch["formula"], batch["mask"]
        binned_ms = batch["binned_MS"]

        FP_pred, binned_ms_pred = model(intensities, formula, mask, binned_ms)

        return FP_pred, binned_ms_pred

    def train_loss_on_outputs(self, outputs, batch):
        
        FP_pred, binned_ms_pred = outputs
        FP = batch["FP"]
        binned_ms = batch["binned_MS"]
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + reconstruction_loss

    def train_regularization(self, params):
        return 0.01 * torch.square(params.norm())

    def test_loss(self, model, params, batch):

        # Unpack the batch 
        intensities, formula, mask = batch["intensities"], batch["formula"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        FP_pred, binned_ms_pred = model(intensities, formula, mask, binned_ms)

        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + reconstruction_loss

class ObjectiveFrags(BaseObjective):

    def train_outputs(self, model, batch):

        # Unpack the batch 
        intensities, mask = batch["intensities"], batch["mask"]
        frags_tokens, frags_mask, frags_weight = batch["frags_tokens"], batch["frags_mask"], batch["frags_weight"]
        binned_ms = batch["binned_MS"]

        FP_pred, binned_ms_pred = model(intensities, mask, binned_ms, frags_tokens, frags_mask, frags_weight)

        return FP_pred, binned_ms_pred

    def train_loss_on_outputs(self, outputs, batch):
        
        FP_pred, binned_ms_pred = outputs
        FP = batch["FP"]
        binned_ms = batch["binned_MS"]
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + reconstruction_loss

    def train_regularization(self, params):
        return 0.01 * torch.square(params.norm())

    def test_loss(self, model, params, batch):

        # Unpack the batch 
        intensities, mask = batch["intensities"], batch["mask"]
        frags_tokens, frags_mask, frags_weight = batch["frags_tokens"], batch["frags_mask"], batch["frags_weight"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        FP_pred, binned_ms_pred = model(intensities, mask, binned_ms, frags_tokens, frags_mask, frags_weight)

        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)
        FP_loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        return FP_loss + reconstruction_loss

def get_checkpoint_path(folder):

    checkpoints = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    best_checkpoint, lowest_loss = "", 1e4

    for c in checkpoints:

        loss = float(c.replace(".ckpt", "").split("=")[-1]) # hack 
        if loss < lowest_loss:
            lowest_loss = loss 
            best_checkpoint = c 
    
    return os.path.join(folder, best_checkpoint)

def get_model(model_cache_folder, params):

    model_path = get_checkpoint_path(model_cache_folder)

    model_name = params["model"]["name"] 
    if model_name == "binned_MS_encoder":
        model = MSBinnedModel.load_from_checkpoint(model_path)

    elif model_name == "MS_encoder":
        model = MSTransformerEncoder.load_from_checkpoint(model_path)

    elif model_name == "formula_encoder":
        model = FormulaTransformerEncoder.load_from_checkpoint(model_path)

    elif model_name == "frag_encoder":

        model = FragTransformerEncoder.load_from_checkpoint(model_path)  
        
    else:
        raise NotImplementedError()
    
    return model 

def get_dataloader_dataset(params):

    params["data"]["get_CF"], params["data"]["get_frags"] = False, False
    model_name = params["model"]["name"]

    if model_name == "formula_encoder": params["data"]["get_CF"] = True 
    if model_name == "frag_encoder": params["data"]["get_frags"] = True 

    dataset = MSDataset(**params["data"])
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()

    train_data = dataset.train_data
    test_data = dataset.test_data

    return train_loader, test_loader, train_data, test_data

def get_module(device, model_name, model, train_loader, test_loader):

    if model_name == "binned_MS_encoder":
        objective = ObjectiveBinnedMS()

    elif model_name == "MS_encoder":
        objective = ObjectiveMS()

    elif model_name == "formula_encoder":
        objective = ObjectiveFormula()

    elif model_name == "frag_encoder":
        objective = ObjectiveFrags()

    else:
        raise NotImplementedError()

    module = LiSSAInfluenceModule(model = model, objective = objective,
                                    train_loader = train_loader, test_loader = test_loader,
                                    device = device, damp = 0.001, repeat = 5, depth = 5, scale = 0.001)

    return module

def check_empty(path):
    
    data = load_pickle(path)
    return len(data) == 0 

def get_scores(module, train_idx, test_idx):
    
    all_scores = {} 

    for i_test, id_test in tqdm(test_idx):

        scores = []

        for (i_train, _) in train_idx:
            
            s =  module.influences([i_train], [i_test])
            scores.append(s)

        if sum(scores) == 0: return {} # Something is wrong - return empty and attempt to run again

        all_scores[id_test] = scores

    return all_scores

if __name__ == "__main__":

    top_k = 100

    # Iterate through all models to check for percentage detection
    folder = "./results_cache/"
    all_folders = []
    
    for model in os.listdir(folder):
        subfolder = os.path.join(folder, model)
        for checkpoint in os.listdir(subfolder):
            
            if "cleaned" in checkpoint: continue
            all_folders.append(os.path.join(subfolder, checkpoint))

    # Get the problematic set 
    problem_set_path = "/data/rbg/users/klingmin/projects/MS_processing/data/massspecgym/massspecgym_problem_set.pkl"
    problem_set = [r["id_"] for r in load_pickle(problem_set_path)]
    
    for f in all_folders:
        
        # Get params and models
        params = read_config(os.path.join(f, "run.yaml"))
        model = get_model(f, params)

        # Get the dataloaders and datasets 
        train_loader, test_loader, train_data, test_data = get_dataloader_dataset(params)

        # Identify the list of train and test idx to analyze 
        train_idx_to_analyze = [(i,os.path.basename(f).replace(".pkl", "")) for i, f in enumerate(train_data) \
                                if os.path.basename(f).replace(".pkl", "") in problem_set]


        test_results_path = os.path.join(f, "test_results.pkl")
        test_results = load_pickle(test_results_path)
        test_results = sorted(test_results.items(), key = lambda item: item[1]["loss"], reverse = True)
        test_results_to_analyze = test_results[:top_k]
        test_results_id_ = [r[0] for r in test_results_to_analyze]

        test_idx_to_analyze = [(i,os.path.basename(f).replace(".pkl", "")) for i, f in enumerate(test_data) \
                               if os.path.basename(f).replace(".pkl", "") in test_results_id_]
        
        # Get the module 
        model_name = params["model"]["name"]
        device = torch.device('cuda:1')
        module = get_module(device, model_name, model, train_loader, test_loader)

        # Get the scores 
        output_path = os.path.join(f, "influence_scores.pkl")

        if not os.path.exists(output_path) or check_empty(output_path):
            
            all_scores = get_scores(module, train_idx_to_analyze, test_idx_to_analyze)
            pickle_data(all_scores, output_path)
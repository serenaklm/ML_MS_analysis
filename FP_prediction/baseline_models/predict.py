import os
import copy
import argparse
import numpy as np 
from tqdm import tqdm 

import torch
import torch.nn.functional as F 

from modules import *
from dataloader import MSDataset

from utils import pickle_data, write_json, read_config, load_pickle

def to_binary(FP, threshold):

    FP = torch.sigmoid(FP).cpu().numpy()
    FP = (FP > threshold).astype(int)

    return FP 

@torch.no_grad()
def get_loss(FP_pred, FP):
    return F.binary_cross_entropy_with_logits(FP_pred, FP, reduce = False)

@torch.no_grad()
def batch_jaccard_index(FP_pred, FP):

    # Intersection = bitwise AND
    intersection = np.logical_and(FP, FP_pred).sum(axis=1)

    # Union = bitwise OR
    union = np.logical_or(FP, FP_pred).sum(axis=1)

    # Avoid division-by-zero by adding a small epsilon
    jaccard_scores = intersection / (union + 1e-9)
    total_jaccard = jaccard_scores.sum()

    return total_jaccard

@torch.no_grad()
def forward(model_name, model, batch, device, include_adduct, include_CE, include_instrument):

    adduct, CE, instrument = None, None, None
    if include_adduct: adduct = batch["adduct"].to(device)
    if include_CE: CE = batch["CE"].to(device)
    if include_instrument: instrument = batch["instrument"].to(device)
    
    if model_name == "binned_MS_encoder":
        
        FP_pred, _ = model(batch["binned_MS"].to(device), adduct, CE, instrument)

    elif model_name == "MS_encoder":

        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]

        mz, intensities, mask = mz.to(device), intensities.to(device), mask.to(device)
        binned_ms = binned_ms.to(device)
        FP_pred, _ = model(mz, intensities, mask, binned_ms, adduct, CE, instrument)

    elif model_name == "formula_encoder":

        intensities, formula, mask = batch["intensities"], batch["formula"], batch["mask"]
        binned_ms = batch["binned_MS"]

        intensities, formula, mask = intensities.to(device), formula.to(device), mask.to(device)
        binned_ms = binned_ms.to(device)

        FP_pred, _ = model(intensities, formula, mask, binned_ms, adduct, CE, instrument)

    elif model_name == "frag_encoder":

        intensities, mask = batch["intensities"], batch["mask"]
        frags_tokens, frags_mask, frags_weight = batch["frags_tokens"], batch["frags_mask"], batch["frags_weight"]
        binned_ms = batch["binned_MS"]

        intensities, mask = intensities.to(device), mask.to(device)
        frags_tokens, frags_mask, frags_weight  = frags_tokens.to(device), frags_mask.to(device), frags_weight.to(device)  
        binned_ms = binned_ms.to(device)

        FP_pred, _ = model(intensities, mask, binned_ms, frags_tokens, frags_mask, frags_weight)

    else:
        raise Exception() 
    
    return FP_pred 

@torch.no_grad()
def predict(model, config, device, batch_size, threshold = 0.5):

    model_name = config["model"]["name"] 
    get_CF, get_frags = False, False

    if model_name == "formula_encoder": get_CF = True 
    if model_name == "frag_encoder": get_frags = True 

    # Update the data directory 
    config["data"]["dir"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "frags_preds")
    config["data"]["split_file"] = os.path.join(config["data"]["splits_folder"], config["data"]["dataset"], "splits", config["data"]["split_file"])
    config["data"]["adduct_file"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "all_adducts.pkl")
    config["data"]["instrument_file"] = os.path.join(config["data"]["data_folder"], config["data"]["dataset"], "all_instruments.pkl")

    # Check if we are getting the meta information 
    feats_params = config["model"]["feats_params"]
    include_adduct, include_CE, include_instrument = feats_params["include_adduct"], feats_params["include_CE"], feats_params["include_instrument"]

    dataset = MSDataset(dir = config["data"]["dir"],
                        split_file = config["data"]["split_file"],
                        adduct_file = config["data"]["adduct_file"],
                        instrument_file = config["data"]["instrument_file"],
                        batch_size = batch_size,
                        num_workers = 4,
                        max_da = config["data"]["max_da"], 
                        max_MS_peaks = config["data"]["max_MS_peaks"],
                        bin_resolution = config["data"]["bin_resolution"], 
                        FP_type = config["data"]["FP_type"],
                        intensity_type = config["data"]["intensity_type"], 
                        intensity_threshold = config["data"]["intensity_threshold"],
                        considered_atoms = config["data"]["considered_atoms"],
                        n_frag_candidates = config["data"]["n_frag_candidates"],
                        chemberta_model = config["data"]["chemberta_model"],
                        return_id_ = True, 
                        get_CF = get_CF,
                        get_frags = get_frags)

    data_loader = dataset.test_dataloader()
    
    # Run model predictions
    id_list, predictions, GT, losses = [], [], [], []
    total_loss, total_jaccard, total = 0,0, 0

    for batch in tqdm(data_loader):
        
        # Unpack the batch 
        id_ = batch["id_"]
        FP = batch["FP"]

        # Forward pass
        FP_pred = forward(model_name, model, batch, device, include_adduct, include_CE, include_instrument)
        FP_pred = FP_pred.cpu()

        # Get the loss 
        loss = get_loss(FP_pred, FP)
        loss = loss.mean(-1)
        total_loss += loss.mean(-1).item() * FP_pred.size(0)
        total_jaccard += batch_jaccard_index(to_binary(FP_pred, threshold), FP.numpy()) 
        total += FP_pred.size(0)

        # Save the predctions 
        id_list.extend(id_)
        predictions.append(FP_pred)
        GT.append(FP)
        losses.extend(loss.numpy().tolist())

    # Format the predictions
    predictions = torch.cat(predictions, dim = 0).numpy().tolist()
    GT = torch.cat(GT, dim = 0).numpy().tolist()
    predictions = {id_list[i]: {"pred": predictions[i], "GT": GT[i], "loss": losses[i]} for i in range(len(id_list))}
    
    # Get the average loss 
    avg_loss = total_loss / total 

    # Get the average jaccard loss 
    avg_jaccard = total_jaccard / total

    return predictions, avg_loss, avg_jaccard

def get_checkpoint_path(folder):

    checkpoints = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    best_checkpoint, lowest_loss = "", 1e4

    for c in checkpoints:

        loss = float(c.replace("-v1", "").replace(".ckpt", "").split("=")[-1]) # hack 
        if loss < lowest_loss:
            lowest_loss = loss 
            best_checkpoint = c 
    
    return os.path.join(folder, best_checkpoint)

def main(args):

    # Get the checkpoint and config
    checkpoint_dir = args.checkpoint 
    config = read_config(os.path.join(checkpoint_dir, "run.yaml"))

    # Load the model
    model_name = config["model"]["name"]  
    
    if model_name == "binned_MS_encoder":
        model = MSBinnedModel.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))

    elif model_name == "MS_encoder":
        model = MSTransformerEncoder.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))

    elif model_name == "formula_encoder":
        model = FormulaTransformerEncoder.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))

    elif model_name == "frag_encoder":

        model = FragTransformerEncoder.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))  
    else:
        raise NotImplementedError()

    model.eval()
    model.to(args.device)

    # Get the predictions 
    predictions, loss, jaccard = predict(model, config, args.device, args.batch_size)
     
    # Write the predictions
    output_path = os.path.join(checkpoint_dir, "test_results.pkl")
    pickle_data(predictions, output_path)
    write_json({"loss": loss, "jaccard": jaccard}, os.path.join(checkpoint_dir, "test_performance.json"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type = int, default = 512, help = "Batch size when running prediction.")
    parser.add_argument("--device", type = str, default = "cuda", help = "The device to use for prediction.")
    parser.add_argument("--checkpoint", type = str, help = "Path to a model checkpoint")
    args = parser.parse_args()

    # Manually add in (hack)
    # folder = "./models_cached/"
    # all_folders = []
    
    # for FP in os.listdir(folder):
    #     FP_folder = os.path.join(folder, FP)
    #     for model in os.listdir(FP_folder):
    #         model_folder = os.path.join(FP_folder, model)
    #         for checkpoint in os.listdir(model_folder):
                
    #             all_folders.append(os.path.join(model_folder, checkpoint))

    # Manually add in (hack)
    folder = "./best_models/"
    all_folders = []
    
    for model in os.listdir(folder):
        model_folder = os.path.join(folder, model)
        for checkpoint in os.listdir(model_folder):
            all_folders.append(os.path.join(model_folder, checkpoint))

    # # # Manually add in (hack)
    # folder = "./results/"
    # all_folders = []

    # for model in os.listdir(folder):
    #     for checkpoint in os.listdir(os.path.join(folder, model)):
    #         all_folders.append(os.path.join(folder, model, checkpoint))

    for f in all_folders:

        args.checkpoint = f
        check_test_performance = os.path.exists(os.path.join(f, "test_performance.json"))
        check_test_results = os.path.exists(os.path.join(f, "test_results.pkl"))

        if check_test_performance:
            assert check_test_results
            continue 

        print("Running prediction for: ", f)
        main(args)
        print("Prediction complete")
    
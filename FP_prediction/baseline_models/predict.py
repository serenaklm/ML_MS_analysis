import os
import argparse
import numpy as np 
from tqdm import tqdm 

import torch
import torch.nn.functional as F 

from dataloader import MSDataset
from modules import MSBinnedModel

from utils import pickle_data, write_json, read_config

@torch.no_grad()
def get_loss(FP_pred, FP):
    return F.binary_cross_entropy_with_logits(FP_pred, FP)

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
def predict(model, config, device, batch_size):

    bin_resolution = config["data"]["bin_resolution"]
    max_da = config["data"]["max_da"]
    FP_type = config["data"]["FP_type"]

    dataset = MSDataset(dir = config["data"]["dir"],
                        test_folder = config["data"]["test_folder"],
                        batch_size = batch_size,
                        bin_resolution = bin_resolution,
                        max_da = max_da, 
                        FP_type = FP_type,
                        num_workers = 4, 
                        intensity_type = "", 
                        intensity_threshold = 0.0,
                        mode = "inference")
    
    data_loader = dataset.test_dataloader()
    
    # Run model predictions
    id_list, predictions, GT = [], [], []
    total_loss, total_jaccard, total = 0,0, 0

    for batch in tqdm(data_loader):
        
        # Unpack the batch 
        id_ = batch["id_"]
        binned_ms = batch["binned_MS"].to(device)
        FP = batch["FP"]

        # Forward pass
        FP_pred = model(binned_ms).cpu()

        # Get the loss 
        total_loss += get_loss(FP_pred, FP).item() * FP_pred.size(0)
        total_jaccard += batch_jaccard_index(FP_pred.cpu().numpy(), FP.numpy())
        total += FP_pred.size(0)

        # Save the predctions 
        id_list.extend(id_)
        predictions.append(FP_pred)
        GT.append(FP)

    # Format the predictions
    predictions = torch.cat(predictions, dim = 0).numpy().tolist()
    GT = torch.cat(GT, dim = 0).numpy().tolist()
    predictions = {id_list[i]: {"pred": predictions[i], "GT": GT[i]} for i in range(len(id_list))}
    
    # Get the average loss 
    avg_loss = total_loss / total 

    # Get the average jaccard loss 
    avg_jaccard = total_jaccard / total

    return predictions, avg_loss, avg_jaccard

def main(args):

    # Get the checkpoint and config
    checkpoint_dir = args.checkpoint 
    config = read_config(os.path.join(checkpoint_dir, "run.yaml"))

    # Load the model 
    model = MSBinnedModel.load_from_checkpoint(os.path.join(checkpoint_dir, "model.ckpt"))
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

    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size when running prediction.")
    parser.add_argument("--device", type = str, default = "cuda", help = "The device to use for prediction.")
    parser.add_argument("--checkpoint", type = str, help = "Path to a model checkpoint")
    args = parser.parse_args()

    # Manually add in (hack)
    folder = "./results/"
    all_folders = [os.path.join(folder, f) for f in os.listdir(folder)]

    for idx, f1 in enumerate(all_folders):    
        all_folders[idx] = os.path.join(f1, [f for f in os.listdir(f1)][0])
        
    for f in all_folders:

        args.checkpoint = f
        print("Running prediction for: ", f)
        main(args)
        print("Prediction complete")
    
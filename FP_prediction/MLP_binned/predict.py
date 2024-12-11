import os
import argparse
from tqdm import tqdm 

import torch
import torch.nn.functional as F 

from dataloader import BinnedMSPredDataset
from modules import MSBinnedModel

from utils import pickle_data, read_config

@torch.no_grad()
def predict(model, config, device, batch_size):

    bin_resolution = config["data"]["bin_resolution"]
    max_da = config["data"]["max_da"]
    FP_type = config["data"]["FP_type"]

    filepath = os.path.join(config["data"]["dir"], "test_data.pkl")

    dataset = BinnedMSPredDataset(filepath, 
                                  batch_size = batch_size,
                                  bin_resolution = bin_resolution,
                                  max_da = max_da, 
                                  FP_type = FP_type,
                                  num_workers = 4)
    
    data_loader = dataset.test_dataloader()
    
    # Run model predictions
    id_list, predictions = [], []

    for batch in tqdm(data_loader):
        
        # Unpack the batch 
        id_ = batch["new_id_"]
        binned_ms = batch["binned_MS"].to(device)

        # Forward pass
        FP_pred = model(binned_ms).cpu()

        # Save the predctions 
        id_list.extend(id_)
        predictions.append(FP_pred)

    # Save predictions
    predictions = torch.cat(predictions, dim = 0).numpy().tolist()
    predictions = {id_list[i]: predictions[i] for i in range(len(id_list))}

    return predictions

def main(args):

    # Get the checkpoint and config
    checkpoint_dir = args.checkpoint 
    config = read_config(os.path.join(checkpoint_dir, "run.yaml"))

    # Load the model 
    model = MSBinnedModel.load_from_checkpoint(os.path.join(checkpoint_dir, "model.ckpt"))
    model.eval()
    model.to(args.device)

    # Get the predictions 
    FP_pred = predict(model, config, args.device, args.batch_size)

    # Get the F score 
    

    # Write the predictions
    output_path = os.path.join(checkpoint_dir, "test_results.pkl")
    pickle_data(FP_pred, output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size when running prediction.")
    parser.add_argument("--device", type = str, default = "cuda", help = "The device to use for prediction.")
    parser.add_argument("--checkpoint", type = str, help = "Path to a model checkpoint")
    args = parser.parse_args()

    # Manually add in (hack)
    folder = "./results"
    all_folders = [os.path.join(folder, f) for f in os.listdir(folder)]

    for idx, f1 in enumerate(all_folders):    
        all_folders[idx] = os.path.join(f1, [f for f in os.listdir(f1)][0])
        
    for f in all_folders:

        args.checkpoint = f
        main(args)

    
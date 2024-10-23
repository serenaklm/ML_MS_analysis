import os
import argparse
from tqdm import tqdm 

import torch
import torch.nn.functional as F 

from dataloader import BinnedMSPredDataset
from modules import MSBinnedModel

from utils import pickle_data

@torch.no_grad()
def predict(filepath, model, device, batch_size):

    dataset = BinnedMSPredDataset(filepath, batch_size = batch_size, num_workers = 4)
    data_loader = dataset.test_dataloader()
    
    # Run model predictions
    id_list, predictions = [], []

    for batch in tqdm(data_loader):
        
        # Unpack the batch 
        id_ = batch["new_id_"]
        binned_ms = batch["binned_MS"].to(device)
        adduct_idx, instrument_idx = batch["adduct_idx"].to(device), batch["instrument_idx"].to(device)

        # Forward pass
        FP_pred = model(binned_ms, adduct_idx, instrument_idx).cpu()

        # Save the predctions 
        id_list.extend(id_)
        predictions.append(FP_pred)

    # Save predictions
    predictions = torch.cat(predictions, dim = 0).numpy().tolist()
    predictions = {id_list[i]: predictions[i] for i in range(len(id_list))}

    return predictions

def main(args):

    # Get the config
    checkpoint_dir = args.checkpoint 

    # Load the model 
    model = MSBinnedModel.load_from_checkpoint(os.path.join(checkpoint_dir, "model.ckpt"))
    model.eval()
    model.to(args.device)

    # Get the predictions 
    FP_pred = predict(args.test_filepath, model, args.device, args.batch_size)

    # Write the predictions
    output_path = os.path.join(checkpoint_dir, "test_results.pkl")
    pickle_data(FP_pred, output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size when running prediction.")
    parser.add_argument("--test_filepath", type = str, help = "Filepath of the test data requirede for prediction.")
    parser.add_argument("--device", type = str, default = "cuda", help = "The device to use for prediction.")
    parser.add_argument("--checkpoint", type = str, required = True, help = "Path to a model checkpoint")
    args = parser.parse_args()
    main(args)
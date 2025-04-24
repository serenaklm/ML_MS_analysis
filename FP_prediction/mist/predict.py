import os
import argparse
import numpy as np 
from tqdm import tqdm 

import torch 
import torch.nn.functional as F 

from utils import read_config, pickle_data, write_json
from mist.data import datasets, splitter, featurizers

from model.mist_model import MistNet

def to_binary(FP, threshold):

    FP = FP.cpu().numpy()
    FP = (FP > threshold).astype(int)

    return FP 

@torch.no_grad()
def get_loss(FP_pred, FP):
    return F.binary_cross_entropy(FP_pred, FP, reduce = False)

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

def get_checkpoint_path(folder):

    checkpoints = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    best_checkpoint, lowest_loss = "", 1e4

    for c in checkpoints:

        loss = float(c.replace("-v1", "").replace(".ckpt", "").split("=")[-1]) # hack 
        if loss < lowest_loss:
            lowest_loss = loss 
            best_checkpoint = c 
    
    return os.path.join(folder, best_checkpoint)

def get_datamodule(config):

    # Split data
    my_splitter = splitter.get_splitter(**config["dataset"])

    # Update the config now 
    config["dataset"]["spec_features"] = "peakformula_test"
    config["dataset"]["allow_none_smiles"] = False

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**config["dataset"])

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**config["dataset"])
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Get the test split 
    _, (_, _, test) = my_splitter.get_splits(spectra_mol_pairs)

    test_dataset = datasets.SpectraMolDataset(spectra_mol_list=test, featurizer=paired_featurizer, **config["train_settings"])
    test_loader = datasets.SpecDataModule.get_paired_loader(test_dataset, shuffle=False)

    return test_loader

def batch_to_device(batch: dict, device) -> None:
    
    """batch_to_device.

    Convert batch tensors to same device as the model


    Args:
        batch (dict): Batch from data loader

    """
    # Port to cuda
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = batch[key].to(device)

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

@torch.no_grad()
def predict(model, config, device, threshold = 0.5):

    # Get the dataset
    test_loader = get_datamodule(config)

    # Run model predictions
    id_list, predictions, GT, losses = [], [], [], []
    total_loss, total_jaccard, total = 0,0, 0

    for spectra_batch in tqdm(test_loader):

        id_ = spectra_batch["names"]
        FP = spectra_batch["mols"][:, :].to(float)

        batch_to_device(spectra_batch, device)

        # Get the predicted fingerprints 
        FP_pred = model.encode_spectra(spectra_batch)[0].to(float).cpu()

        # Get the loss 
        loss = get_loss(FP_pred, FP)
        loss = loss.mean(-1)
        total_loss += loss.mean(-1).item() * FP_pred.size(0)
        total_jaccard += batch_jaccard_index(to_binary(FP_pred, threshold), FP.numpy().astype(int)) 
        total += FP_pred.size(0)
        print(total_jaccard, total)

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

def main(args):

    # Get the checkpoint and config
    checkpoint_dir = args.checkpoint 
    config = read_config(os.path.join(checkpoint_dir, "run.yaml"))
    config = update_config(args, config)

    # Get the model 
    model = MistNet.load_from_checkpoint(get_checkpoint_path(checkpoint_dir))
    model.eval()
    model.to(args.device)

    # Get the predictions
    predictions, loss, jaccard = predict(model, config, args.device)

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
    folder = "./best_models/nist2023_sieved"
    all_folders = []
    
    # for FP in os.listdir(folder):
    #     FP_folder = os.path.join(folder, FP)
    #     for dataset in os.listdir(FP_folder):
    #         dataset_folder = os.path.join(FP_folder, dataset)
    #         for checkpoint in os.listdir(dataset_folder):
                
    #             all_folders.append(os.path.join(dataset_folder, checkpoint))

    # for split in os.listdir(folder):
    #     split_folder = os.path.join(folder, split)
    for checkpoint in os.listdir(folder):
        all_folders.append(os.path.join(folder, checkpoint))


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
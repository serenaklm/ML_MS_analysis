import copy
import argparse 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.data import Dataset, Subset, DataLoader, random_split, RandomSampler

from dataloader import get_dataloader
from utils import print, to_tensor, get_optim, optim_step

@torch.no_grad()
def test_predictor(data: Dataset = None,
                   loader: DataLoader = None,
                   test_indices: list[int] = None,
                   predictor: torch.nn.Module = None,
                   config_dict: dict = None):
    
    """
        Apply the predictor to the test sets
    """

    predictor.eval()

    # If the data loader is not provided, create a data loader from the data.
    if loader is None:
        assert data is not None, "data and loader cannot both be None"

        if test_indices is None: test_data = data
        else: test_data = Subset(data, indices = test_indices)

        loader = get_dataloader(test_data, shuffle = False, batch_size = config_dict["batch_size"], 
                                num_workers = config_dict["num_workers"], config_dict = config_dict)
        
    
    # Get the jaccard index metric 
    metric = BinaryJaccardIndex(threshold = config_dict["class_threshold"]).to(config_dict["device"])

    total, count = 0, 0

    for mz_binned, FP in loader:
        
        FP = to_tensor(FP).to(config_dict["device"])
        FP_pred = predictor(mz_binned)
        jaccard_similarity = metric(FP_pred, FP).detach().cpu().item()

        total += jaccard_similarity
        count += 1
        
    return total / count

def train_predictor(data: Dataset = None,
                    train_indices: list[int] = None,
                    predictor: torch.nn.Module = None,
                    config_dict: dict = None):
    
    """
        Train the predcitor on train split
        We sample a random set from train for early stopping and validation
    """

    train_data = Subset(data, indices = train_indices)

    # Randomly sample a validation set (size 1/3) from the train split.
    # We will monitor the performance on the validation data to prevent memorization.

    train_size = int(len(train_data) // 3 * 2)
    val_size   = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_loader = get_dataloader(train_data, False, config_dict["batch_size"], config_dict["num_workers"], 
                                  config_dict = config_dict,
                                  sampler = RandomSampler(train_data, replacement = True,
                                                          num_samples = config_dict["batch_size"] * config_dict["num_batches"]))

    val_loader = get_dataloader(val_data, False, config_dict["batch_size"], config_dict["num_workers"],
                                config_dict = config_dict,
                                sampler = RandomSampler(val_data, replacement = True,
                                                        num_samples = config_dict["batch_size"] * config_dict["num_batches"])) 

    # Get the optimizer and loss for the predictor
    opt = get_optim(predictor, config_dict)

    # Start training. Terminate training if the validation accuracy stops improving
    best_val_score, best_predictor = -1, None
    ep, cycle = 0, 0

    # Get the loss
    ce_loss = nn.CrossEntropyLoss()

    while True:

        predictor.train()

        for mz_binned, FP in train_loader:

            FP = to_tensor(FP).to(config_dict["device"])
            FP_pred = predictor(mz_binned)

            loss = ce_loss(FP_pred, FP)
            optim_step(predictor, opt, loss, config_dict)

        val_score = test_predictor(loader = val_loader, 
                                   predictor = predictor, config_dict = config_dict)

        progress_message =  f'Train predictor @ epoch: {ep}, mean val jaccard similarity: {val_score:.2f}'
        print(progress_message, end="\r", flush=True, time=True)

        if val_score > best_val_score:
            best_val_score = val_score
            best_predictor = copy.deepcopy(predictor.state_dict())
            cycle = 0
        
        else: cycle += 1
        if cycle == config_dict["patience"]: break

        ep += 1

    # Load the best predictor
    predictor.load_state_dict(best_predictor)

    # Clear the progress
    # Note that we need to overwrite the time stamps too.
    print(" " * (20 + len(progress_message)), end="\r", time=False)

    return best_val_score
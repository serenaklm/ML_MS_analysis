import copy
from tqdm import tqdm
from typing import List 

import torch
import torch.nn.functional as F 
from torch.utils.data import random_split, Dataset, Subset, DataLoader, RandomSampler

from utils import get_optim, optim_step, print

@torch.no_grad()
def test_predictor(data: Dataset = None,
                   loader: DataLoader = None,
                   test_indices: List[int] = None, 
                   predictor: torch.nn.Module = None,
                   config: dict = None):

    predictor.eval()

    # If the data loader is not provided, create a data loader from the data.
    if loader is None:

        assert data is not None, "data and loader cannot both be None"

        if test_indices is None:
            test_data = data
        else:
            test_data = Subset(data, indices = test_indices)

        batch_size = config["dataprocessing"]["batch_size"]
        num_batches = config["dataprocessing"]["num_batches"]
        num_workers = config["dataprocessing"]["num_workers"]
        
        loader = DataLoader(test_data, batch_size = batch_size,
                            shuffle = False, num_workers = num_workers)

    total, count = 0, 0

    for batch in tqdm(loader):
        
        FP = batch["FP"]
        FP_pred = predictor(batch)
        loss = F.binary_cross_entropy_with_logits(FP_pred, FP)

        total += loss.item()
        count += 1
    
    return total / count

def train_predictor(data: Dataset = None,
                    train_indices: List[int] = None, 
                    predictor: torch.nn.Module = None,
                    config: dict = None):

    # Get data 
    train_data = Subset(data, indices = train_indices)
    train_size = int(len(train_data) // 3 * 2)
    val_size   = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # Get dataloaders
    batch_size = config["dataprocessing"]["batch_size"]
    num_batches = config["dataprocessing"]["num_batches"]
    num_workers = config["dataprocessing"]["num_workers"]
    
    train_loader = DataLoader(train_data, batch_size = batch_size,
                              sampler = RandomSampler(train_data, replacement = True,
                                                      num_samples = batch_size * num_batches),
                              num_workers = num_workers)

    val_loader = DataLoader(val_data, batch_size = batch_size,
                            num_workers = num_workers)
    
    # Get the optimizer and loss for the predictor
    opt = get_optim(predictor, config)

    # Start training. Terminate training if the validation accuracy stops improving.
    best_val_score, best_predictor = -1, None
    ep, cycle = 0, 0

    while True:

        predictor.train()

        for batch in tqdm(train_loader):
            
            FP = batch["FP"]
            FP_pred = predictor(batch)
            loss = F.binary_cross_entropy_with_logits(FP_pred, FP)
            optim_step(predictor, opt, loss, config)

        # Validate 
        val_score = test_predictor(predictor = predictor,
                                   loader = val_loader,
                                   config = config)

        progress_message =  f'train predictor ep {ep}, val cosine similarity score {val_score:.2f}'
        print(progress_message, end="\r", flush=True, time=True)

        if val_score > best_val_score:
            best_val_score = val_score
            best_predictor = copy.deepcopy(predictor.state_dict())
            cycle = 0
        else:
            cycle += 1

        if cycle == config["model"]["train_params"]["patience"]:
            break
        
        ep += 1

    # Load the best predictor
    predictor.load_state_dict(best_predictor)

    # Clear the progress
    print(" " * (20 + len(progress_message)), end = "\r", time = False)

    return best_val_score

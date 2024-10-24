from typing import List
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical


@torch.no_grad()
def split_data(data: Dataset = None,
               splitter: torch.nn.Module = None,
               config: dict = None,
               random_split = False):
    
    splitter.eval()

    dataloader = DataLoader(data, shuffle = False,
                             batch_size = config["dataprocessing"]["batch_size"],
                             num_workers = config["dataprocessing"]["num_workers"])
    
    total_mask, total_y = [], []

    for batch in dataloader:

        # Split each batch into train split and test split
        # do random split at the start of ls
        if random_split:
        
            train_ratio = config["model"]["train_params"]["train_ratio"]
            prob = torch.ones(len(batch["FP"])).unsqueeze(-1)
            prob = torch.cat([prob * (1 - train_ratio),  # test split prob
                              prob * train_ratio], dim = -1)  # train split prob
            
        else:
            logits = splitter(batch)
            prob = F.softmax(logits, dim = -1)

        # Sample the binary mask
        # 0: test split, 1: train split
        sampler = Categorical(prob)
        mask = sampler.sample()
        mask = mask.long()
        total_mask.append(mask.cpu())
    
    total_mask = torch.cat(total_mask)

    train_indices = total_mask.nonzero().squeeze(1).tolist()
    test_indices = (1 - total_mask).nonzero().squeeze(1).tolist()

    # Get some stats 
    split_stats = {}
    split_stats["train_size"] = len(train_indices)
    split_stats["test_size"] = len(test_indices)
    split_stats["train_ratio"] = len(train_indices) / len(total_mask) * 100
    split_stats["test_ratio"] = len(test_indices) / len(total_mask) * 100

    return split_stats, train_indices, test_indices
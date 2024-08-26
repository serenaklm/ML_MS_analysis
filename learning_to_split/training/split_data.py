import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

from utils import *
from dataloader import get_dataloader

def split_data(data: Dataset = None,
               splitter: torch.nn.Module = None,
               config_dict: dict = None,
               random_split = False):
    
    """
        Sample the splitting decisions from the Splitter.
        If random_split positive, apply random split instead.    
    """

    splitter.eval()

    # Load the data in testing mode (no shuffling so that we can keep track of
    # the index of each example)

    total_mask, total_y = [], []

    # Printing of statistics 
    progress_message = "split_data"
    print(progress_message, end = "\r", flush = True, time = True)

    with torch.no_grad():

        test_loader = get_dataloader(data, shuffle = False, 
                                     batch_size = config_dict["batch_size"], 
                                     num_workers = config_dict["num_workers"], 
                                     config_dict = config_dict)
        
        for mz_binned, _ in test_loader:

            # Get the probability of being either in the train or the test
            if random_split:
                prob = torch.ones(len(mz_binned)).unsqueeze(-1)

                prob = torch.cat([prob * (1 - config_dict["train_ratio"]),  # test split prob
                                  prob * config_dict["train_ratio"]], dim = -1)  # train split prob
            else:
                raise NotImplementedError() 

            # Sample the binary mask
            sampler = Categorical(prob)
            mask = sampler.sample()
            mask = mask.long()

            total_mask.append(mask.cpu())

        # Aggregate all splitting decisions
        total_mask = torch.cat(total_mask)

        # Gather the training indices (indices with mask = 1)
        # and the testing indices (indices with mask = 0)
        train_indices = total_mask.nonzero().squeeze(1).tolist()
        test_indices = (1 - total_mask).nonzero().squeeze(1).tolist()

        # Compute the statistics of the current split
        split_stats = {}

        split_stats['train_size'] = len(train_indices)
        split_stats['test_size'] = len(test_indices)
        split_stats['train_ratio'] = len(train_indices) / len(total_mask) * 100
        split_stats['test_ratio'] = len(test_indices) / len(total_mask) * 100

        print(" " * (20 + len(progress_message)), end = "\r", time = False)

        return split_stats, train_indices, test_indices
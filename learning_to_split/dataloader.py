import os 
import numpy as np 
from utils import get_all_spectra

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class CustomedDataset(Dataset):
    
    def __init__(self, filepath):

        self.data = get_all_spectra(filepath)
    
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def get_DDP_dataloader(rank, world_size, folder, batch_size, seed, pin_memory = False, num_workers = 0, shuffle = True):

    dataset = CustomedDataset(folder)
    sampler = DistributedSampler(dataset, seed = seed, num_replicas = world_size, rank = rank, shuffle = shuffle, drop_last = False)

    dataloader = DataLoader(dataset, batch_size = batch_size,
                                     pin_memory = pin_memory,
                                     num_workers = num_workers,
                                     drop_last = False, 
                                     sampler = sampler)
    return dataloader
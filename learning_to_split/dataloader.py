import os 
import numpy as np 

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class CustomedDataset(Dataset):
    
    def __init__(self):
        raise NotImplementedError() 
    
    def __len__(self):
        raise NotImplementedError()
        
    def __getitem__(self, idx):
        raise NotImplementedError()
    
def collate_fn():

    raise NotImplementedError() 

def get_DDP_dataloader(rank, world_size, folder, batch_size, seed, pin_memory = False, num_workers = 0, shuffle = True):

    dataset = CustomedDataset(folder)
    sampler = DistributedSampler(dataset, seed = seed, num_replicas = world_size, rank = rank, shuffle = shuffle, drop_last = False)

    dataloader = DataLoader(dataset, batch_size = batch_size,
                                     collate_fn = collate_fn,
                                     pin_memory = pin_memory,
                                     num_workers = num_workers,
                                     drop_last = False, 
                                     sampler = sampler)
    return dataloader
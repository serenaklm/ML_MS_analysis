import os 
import numpy as np 
from functools import partial

from utils import get_all_spectra, bin_MS

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class CustomedDataset(Dataset):
    
    def __init__(self, filepath, config_dict):

        data = []

        for rec in get_all_spectra(filepath): 
            
            mz_binned = bin_MS(rec.mz, rec.intensities, max_mass = config_dict["max_mass"], 
                                                        granularity = config_dict["granularity"]) 
            rec.set("mz_binned", mz_binned)
            data.append(rec)

        self.data = data 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(data, FP_type):
    
    mz_binned_lst, FP_lst = [], []

    for rec in data:

        mz_binned_lst.append(rec.metadata["mz_binned"])
        FP_lst.append([int(c) for c in rec.metadata[FP_type]])

    return mz_binned_lst, FP_lst
    
def get_dataloader(data, shuffle, batch_size, num_workers, config_dict, sampler = None):

    # Get partial function 
    collate_fn_p = partial(collate_fn, FP_type = config_dict["FP_type"])

    if sampler is None: 
        loader = DataLoader(data, shuffle = shuffle,
                            batch_size = batch_size,
                            num_workers = num_workers, 
                            collate_fn = collate_fn_p)

    else: 
        loader = DataLoader(data, 
                            batch_size = batch_size,
                            sampler = sampler,
                            num_workers = num_workers, 
                            collate_fn = collate_fn_p)
                
    return loader

def get_DDP_dataloader(rank, data, config_dict, pin_memory = False, shuffle = True):

    sampler = DistributedSampler(data, seed = config_dict["seed"], num_replicas = config_dict["world_size"], rank = rank, shuffle = shuffle, drop_last = False)
    collate_fn_p = partial(collate_fn, FP_type = config_dict["FP_type"])

    dataloader = DataLoader(data, batch_size = config_dict["batch_size"],
                                  collate_fn = collate_fn_p,
                                  pin_memory = pin_memory,
                                  num_workers = config_dict["num_workers"],
                                  drop_last = False, 
                                  sampler = sampler)
    return dataloader
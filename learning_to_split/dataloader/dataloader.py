from typing import Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import bin_MS, sort_intensities, pad_mz_intensities


class CustomedDataset(Dataset):
    
    def __init__(self, data: List = None,
                       bin_resolution: float = 1.0,
                       max_da: int = 1000,
                       max_MS_peaks: int = 1000,
                       FP_type: str = "maccs"):
        
        self.data = data
        self.bin_resolution = bin_resolution
        self.max_da = max_da
        self.max_MS_peaks = max_MS_peaks
        self.FP_type = FP_type
    
    def process(self, sample: Any) -> Any:

        # Pad the mz and intensities
        peaks = sample["peaks"]
        mz_o = [p["mz"] for p in peaks]
        intensities_o = [p["intensity_norm"] for p in peaks]
        precursor_mz = float(sample["precursor_MZ_final"])
        mz, intensities = sort_intensities(mz_o, intensities_o, precursor_mz)
        mz, intensities = mz[:self.max_MS_peaks], intensities[:self.max_MS_peaks]
        pad_length = self.max_MS_peaks - len(mz)
        mz, intensities, mask = pad_mz_intensities(mz, intensities, pad_length)

        # Bin the MS 
        binned_MS = bin_MS(mz_o, intensities_o, self.bin_resolution, self.max_da)

        # Get the FP
        FP = [float(c) for c in sample[self.FP_type]]

        return {"mz": torch.tensor(mz, dtype=torch.float),
                "intensities": torch.tensor(intensities, dtype=torch.float),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "binned_MS": torch.tensor(binned_MS, dtype = torch.float),
                "FP": torch.tensor(FP, dtype = torch.float)}
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.process(self.data[idx])

def get_DDP_dataloader(rank, world_size, data, batch_size, seed, 
                       pin_memory = False, num_workers = 0, shuffle = True):

    dataset = CustomedDataset(data)
    sampler = DistributedSampler(dataset, seed = seed, num_replicas = world_size, rank = rank, shuffle = shuffle, drop_last = False)
 
    dataloader = DataLoader(dataset, batch_size = batch_size,
                                     pin_memory = pin_memory,
                                     num_workers = num_workers,
                                     drop_last = False, 
                                     sampler = sampler)
    return dataloader
import os
from typing import Any, Callable, List

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from utils import get_all_spectra, bin_MS

class Data(object):

    def __init__(self, data: Any, process: Callable):
        self.data = data
        self.process = process

    def __getitem__(self, index: int) -> Any:
        return self.process(self.data[index])

    def __len__(self) -> int:
        return len(self.data)
    
class PredictionDataset:

    def __init__(self, data: List,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       pin_memory: bool = True,
                       max_da: int = 1000,
                       bin_resolution: float = 0.01, 
                       FP_type: str = "morgan"):
        
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._data: List = []
        self.bin_resolution = bin_resolution
        self.max_da = max_da
        self.FP_type = FP_type

        # Prepare splits
        self._data = data

        print("Data length: ", len(self._data))

    @property
    def data(self) -> List:
        """The entire data."""
        return self._data
    
    def process(self, sample: Any) -> Any:

        """Processes a single data sample"""

        # Bin the MS 
        binned_MS = bin_MS(sample.mz, sample.intensities, self.bin_resolution, self.max_da)

        # Geet the FP
        FP = [float(c) for c in sample.metadata[self.FP_type]]

        return {"binned_MS": torch.tensor(binned_MS, dtype = torch.float),
                "adduct_idx": torch.tensor(int(sample.metadata["adduct_idx"]), dtype = torch.long),
                "instrument_idx": torch.tensor(int(sample.metadata["instrument_idx"]), dtype = torch.long),
                "FP": torch.tensor(FP, dtype = torch.float)}
        
    def dataloader(self):
        data = Data(self.data, self.process)
        dataloader = DataLoader(data,
                                num_workers = self.num_workers,
                                pin_memory = self.pin_memory,
                                batch_size = self.batch_size,
                                shuffle = True)

        return dataloader
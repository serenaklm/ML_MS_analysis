import os
from typing import Any, Callable, List

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from utils import load_pickle, bin_MS

class Data(object):

    def __init__(self, data: Any, process: Callable):
        self.data = data
        self.process = process

    def __getitem__(self, index: int) -> Any:
        return self.process(self.data[index])

    def __len__(self) -> int:
        return len(self.data)
    
class BinnedMSDataset(pl.LightningDataModule):

    def __init__(self, dir: str, 
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
        self._train: List = []
        self._val: List = []
        self._test: List = []
        self._data: List = []
        self.bin_resolution = bin_resolution
        self.max_da = max_da
        self.FP_type = FP_type

        # Get the data 
        train = load_pickle(os.path.join(dir, "train_data.pkl"))
        val = load_pickle(os.path.join(dir, "val_data.pkl"))
        test = load_pickle(os.path.join(dir, "test_data.pkl"))

        # Prepare splits
        self._data = train + val + test
        self._train = train
        self._val = val
        self._test = test

        print("Train length: ", len(self._train))
        print("Val length: ", len(self._val))
        print("Test length: ", len(self._test))

    @property
    def train_data(self) -> List:
        """The validation data."""
        return self._train

    @property
    def val_data(self) -> List:
        """The validation data."""
        return self._val

    @property
    def test_data(self) -> List:
        """The testing data."""
        return self._test

    def prepare_data(self):
        """Only happens on single GPU, ATTENTION: do no assign states."""
        pass

    def setup(self, stage: str = None):
        """Prepares the data for training, validation, and testing."""
        pass
    
    def process(self, sample: Any) -> Any:

        """Processes a single data sample"""

        # Bin the MS 
        mz = [float(p["mz"]) for p in sample["peaks"]]
        intensities = [float(p["intensity_norm"]) for p in sample["peaks"]]
        binned_MS = bin_MS(mz, intensities, self.bin_resolution, self.max_da)

        # Geet the FP
        FP = [float(c) for c in sample[self.FP_type]]

        return {"binned_MS": torch.tensor(binned_MS, dtype = torch.float),
                "FP": torch.tensor(FP, dtype = torch.float)}
        
    def train_dataloader(self):
        train_data = Data(self.train_data, self.process)
        train_data_loader = DataLoader(train_data,
                                        num_workers = self.num_workers,
                                        pin_memory = self.pin_memory,
                                        batch_size = self.batch_size,
                                        shuffle=True)

        return train_data_loader

    def val_dataloader(self):
        val_data = Data(self.val_data, self.process)
        val_data_loader = DataLoader(val_data,
                                    num_workers = self.num_workers,
                                    pin_memory = self.pin_memory,
                                    batch_size = self.batch_size,
                                    shuffle=False)

        return val_data_loader

    def test_dataloader(self):
        test_data = Data(self.test_data, self.process)
        test_data_loader = DataLoader(test_data,
                                    num_workers = self.num_workers,
                                    pin_memory = self.pin_memory,
                                    batch_size = self.batch_size,
                                    shuffle=False)

        return test_data_loader


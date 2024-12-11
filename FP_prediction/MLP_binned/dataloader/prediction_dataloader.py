import os
from typing import Any, Callable, List

import torch
from torch.utils.data import DataLoader


from utils import get_all_spectra, bin_MS

class Data(object):

    def __init__(self, data: Any, process: Callable):
        self.data = data
        self.process = process

    def __getitem__(self, index: int) -> Any:
        return self.process(self.data[index])

    def __len__(self) -> int:
        return len(self.data)

class BinnedMSPredDataset:
    def __init__(self, filepath,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       bin_resolution: float = 1.0,
                       max_da: int = 1000,
                       FP_type: str = "morgan4_2048",             
                       balance: bool = False,
                       pin_memory: bool = True):
        
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance = balance
        self.pin_memory = pin_memory
        self.bin_resolution = bin_resolution
        self.max_da = max_da
        self.FP_type = FP_type
        self._train: List = []
        self._val: List = []
        self._test: List = []
        self._data: List = []

        # Prepare splits
        self._train = []
        self._val = []
        self._test = get_all_spectra(filepath)
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
        binned_MS = bin_MS(sample.mz, sample.intensities, self.bin_resolution, self.max_da)

        # Get the FP
        FP = [float(c) for c in sample.metadata[self.FP_type]]

        return {"new_id_": sample.metadata["new_id_"],
                "binned_MS": torch.tensor(binned_MS, dtype = torch.float),
                "adduct_idx": torch.tensor(int(sample.metadata["adduct_idx"]), dtype = torch.long),
                "instrument_idx": torch.tensor(int(sample.metadata["instrument_idx"]), dtype = torch.long),
                "FP": torch.tensor(FP, dtype = torch.float)}

    def test_dataloader(self):
        test_data = Data(self.test_data, self.process)
        test_data_loader = DataLoader(test_data,
                                    num_workers = self.num_workers,
                                    pin_memory = self.pin_memory,
                                    batch_size = self.batch_size,
                                    shuffle=False)

        return test_data_loader


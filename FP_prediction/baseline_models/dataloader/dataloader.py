import os
import sys
from typing import Any, Callable, List

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from utils import load_pickle, bin_MS, sort_intensities, pad_mz_intensities

class Data(object):

    def __init__(self, data: Any, process: Callable):
        self.data = data
        self.process = process

    def __getitem__(self, index: int) -> Any:
        return self.process(self.data[index])

    def __len__(self) -> int:
        return len(self.data)
    
class MSDataset(pl.LightningDataModule):

    def __init__(self, dir: str, 
                       train_folder: str = "", 
                       val_folder: str = "",
                       test_folder: str = "",
                       batch_size: int = 32,
                       num_workers: int = 0,
                       pin_memory: bool = True,
                       max_da: int = 1000,
                       max_MS_peaks: int = 100,
                       bin_resolution: float = 0.1, 
                       FP_type: str = "morgan4_2028",
                       intensity_type: str = "raw",
                       intensity_threshold: float = 5.0,
                       mode = "train"):
        
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
        self.max_MS_peaks = max_MS_peaks
        self.FP_type = FP_type
        self.intensity_type = intensity_type
        self.intensity_threshold = intensity_threshold

        # Get the data 
        if mode == "train":
            train = [os.path.join(dir, train_folder, f) for f in os.listdir(os.path.join(dir, train_folder))]
            val = [os.path.join(dir, val_folder, f) for f in os.listdir(os.path.join(dir, val_folder))]
            test = [os.path.join(dir, test_folder, f) for f in os.listdir(os.path.join(dir, test_folder))]

            # Prepare splits
            self._data = train + val + test
            self._train = train
            self._val = val
            self._test = test

            print("Train length: ", len(self._train))
            print("Val length: ", len(self._val))
            print("Test length: ", len(self._test))

        else:
            assert mode == "inference"

            test = [os.path.join(dir, test_folder, f) for f in os.listdir(os.path.join(dir, test_folder))]

            # Prepare splits
            self._train = [] 
            self._val = []
            self._test = test

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

    def _process_intensity(self, intensities_o):
        """Prepares the intensity vector"""

        if self.intensity_type == "raw": 
            return intensities_o
        
        elif self.intensity_type == "binary":
            return [float(i > 0.0) for i in intensities_o]

        elif self.intensity_type == "raw_threshold":
            return [i if i >= self.intensity_threshold else 0.0 for i in intensities_o]

        elif self.intensity_type == "binary_threshold":     
             return [float(i > self.intensity_threshold) for i in intensities_o]
        
        else:
            raise Exception(f"{self.intensity_type} not supported.")
        
    def process(self, filepath: Any) -> Any:

        """Processes a single data sample"""
        sample = load_pickle(filepath)

        # Get the mz and intensities
        peaks = sample["peaks"]
        mz_o = [float(p["mz"]) for p in peaks]
        intensities_o = [float(p["intensity_norm"]) for p in peaks]

        # Add in the precursor_mz in the list of peaks 
        mz_o = [float(sample["precursor_MZ_final_corrected"])] + mz_o
        intensities_o = [100.1] + intensities_o # So that the precursor peak is always at the front

        # Different ways to preprocess the intensity
        intensities_o = self._process_intensity(intensities_o)

        # Pad the MZ and intensities
        mz, intensities = sort_intensities(mz_o, intensities_o)
        mz, intensities = mz[:self.max_MS_peaks], intensities[:self.max_MS_peaks]
        pad_length = self.max_MS_peaks - len(mz)
        mz, intensities, mask = pad_mz_intensities(mz, intensities, pad_length)
    
        # Get the binned MS
        binned_MS = bin_MS(mz, intensities, self.bin_resolution, self.max_da)
    
        # Get the FP
        FP = [float(c) for c in sample[self.FP_type]]

        return {"mz": torch.tensor(mz, dtype=torch.float),
                "intensities": torch.tensor(intensities, dtype=torch.float),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "binned_MS": torch.tensor(binned_MS, dtype = torch.float),
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


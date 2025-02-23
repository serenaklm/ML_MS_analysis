import os
from typing import Any, List
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import load_pickle, bin_MS, \
                  sort_intensities, \
                  pad_mz_intensities, pad_missing_cand, pad_missing_cand_weight, filter_candidates, \
                  process_formula, tokenize_frags

class MSDataset(Dataset):
    
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
                       FP_type: str = "morgan4_2048",
                       intensity_type: str = "raw",
                       intensity_threshold: float = 5.0,
                       considered_atoms: List = ["C", "H", "O", "N"],
                       n_frag_candidates: int = 5,
                       chemberta_model: str = "",
                       return_id_: bool = False,
                       get_CF: bool = False,
                       get_frags: bool = False):
        
        self.dir = dir
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
        self.considered_atoms = considered_atoms
        self.n_frag_candidates = n_frag_candidates
        self.return_id_ = return_id_
        self.get_CF = get_CF
        self.get_frags = get_frags

        # Get the data 

        train = [os.path.join(dir, train_folder, f) for f in os.listdir(os.path.join(dir, train_folder))]
        val = [os.path.join(dir, val_folder, f) for f in os.listdir(os.path.join(dir, val_folder))]
        test = [os.path.join(dir, test_folder, f) for f in os.listdir(os.path.join(dir, test_folder))]

        # Prepare splits
        self._data = train + val + test

        print("Data length: ", len(self._data))

        # Load in the tokenizer 
        self.tokenizer = None
        if self.get_frags:           
            self.tokenizer = AutoTokenizer.from_pretrained(chemberta_model)

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

        # Get the id_ 
        id_ = sample["id_"]

        # Get the mz and intensities
        peaks = sample["peaks"]
        mz_o = [float(p["mz"]) for p in peaks]
        intensities_o = [float(p["intensity_norm"]) for p in peaks]

        # Add in the precursor_mz in the list of peaks 
        mz_o = [float(sample["precursor_MZ_final"])] + mz_o
        intensities_o = [100.1] + intensities_o # So that the precursor peak is always at the front
        intensities_o = self._process_intensity(intensities_o) # Process the intensity differently

        formula_o, frags_o = None, None

        # Get the chemical formula if self.get_CF == True
        if self.get_CF:

            if "nist2023" in self.dir:
                formula_o = [sample["formula"]] + [p["comment"]["f"] for p in peaks]

            else:
                formula_o = [sample["formula"]] + [p["comment"]["f_pred"] for p in peaks]

        # Get possible frags if self.get_frags == True
        if self.get_frags:
            
            frags_o = [(pad_missing_cand(self.n_frag_candidates), pad_missing_cand_weight(self.n_frag_candidates))] + \
                      [filter_candidates(p["comment"]["possible_frags"], self.n_frag_candidates) for p in peaks]

        # Sanity check 
        if len(mz_o) != len(intensities_o):
            raise Exception(f"Lengths do not match. mz: {len(mz_o)}, intensity: {len(intensities_o)}")
        if formula_o is not None and len(mz_o) != len(formula_o):
            raise Exception(f"Lengths do not match. mz: {len(mz_o)}, formula: {len(formula_o)}")
        if frags_o is not None and len(mz_o) != len(frags_o):
            raise Exception(f"Lengths do not match. mz: {len(mz_o)}, frags: {len(frags_o)}")

        # Sort the MZ, intensities, formula and frags
        mz, intensities, formula, frags_smiles, frags_weight = sort_intensities(mz_o, intensities_o, formula_o, frags_o)

        # Get the binned MS 
        binned_MS = bin_MS(mz, intensities, self.bin_resolution, self.max_da)

        # Get subset of the peaks for transformer network 
        mz = mz[:self.max_MS_peaks]
        intensities = intensities[:self.max_MS_peaks]
        if formula is not None: formula = formula[:self.max_MS_peaks]
        if frags_smiles is not None: frags_smiles = frags_smiles[:self.max_MS_peaks]
        if frags_weight is not None: frags_weight = frags_weight[:self.max_MS_peaks]
        pad_length = self.max_MS_peaks - len(mz)

        output = pad_mz_intensities(mz, intensities, formula, frags_smiles, frags_weight, 
                                    pad_length, n_cands = self.n_frag_candidates)
        
        mz, intensities, formula, frags_smiles, frags_weight, mask = output

        # Skip processing some records
        assert sum(mask) != len(mask) and sum(binned_MS) != 0.0

        # Process the formula 
        if formula is not None: formula = [process_formula(f, self.considered_atoms) for f in formula]

        # Process the fragments 
        frags_tokens, frags_mask = None, None
        if frags_smiles is not None: 
            assert frags_weight is not None 
            frags_tokens, frags_mask = tokenize_frags(frags_smiles, self.tokenizer, n_cands = self.n_frag_candidates)
    
        # Get the FP
        FP = [float(c) for c in sample["FPs"][self.FP_type]]

        rec = {"mz": torch.tensor(mz, dtype=torch.float),
               "intensities": torch.tensor(intensities, dtype=torch.float),
               "mask": torch.tensor(mask, dtype=torch.bool),
               "binned_MS": torch.tensor(binned_MS, dtype = torch.float),
               "FP": torch.tensor(FP, dtype = torch.float)}

        if self.return_id_: rec["id_"] = id_
        if formula is not None: rec["formula"] = torch.tensor(formula, dtype=torch.float)
        if frags_tokens is not None: rec["frags_tokens"] = frags_tokens
        if frags_mask is not None: rec["frags_mask"] = frags_mask
        if frags_weight is not None: rec["frags_weight"] = torch.tensor(frags_weight, dtype = torch.float)

        return rec 
    
    def __len__(self):
        return len(self._data)
        
    def __getitem__(self, idx):
        return self.process(self._data[idx])

def get_DDP_dataloader(rank, world_size, data, batch_size, seed, 
                       pin_memory = False, num_workers = 0, shuffle = True):

    dataset = MSDataset(data)
    sampler = DistributedSampler(dataset, seed = seed, num_replicas = world_size, rank = rank, shuffle = shuffle, drop_last = False)
 
    dataloader = DataLoader(dataset, batch_size = batch_size,
                                     pin_memory = pin_memory,
                                     num_workers = num_workers,
                                     drop_last = False, 
                                     sampler = sampler)
    return dataloader
import re
import math
import copy 
import yaml
import json
import pickle
import numpy as np
from tqdm import tqdm
from matchms.importing import load_from_mgf, load_from_msp

import torch 

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def read_config(path):

    with open(path, "r") as f:
        raw_config = yaml.load(f, Loader = yaml.Loader)
        config = copy.deepcopy(raw_config)

    return config

def filter_intensities(mz, intensities, formula):
    mz_f, intensities_f, formula_f = [],[],[]
    for m, i,f in zip(mz, intensities, formula):
        if i > 0.0:
            mz_f.append(m)
            intensities_f.append(i)
            formula_f.append(f)

    return mz_f, intensities_f, formula_f

def sort_intensities(mz, intensities, formula):

    # Filter the mz and intensities
    mz, intensities, formula = filter_intensities(mz, intensities, formula)

    # Sort by intensities
    order = np.argsort(intensities)[::-1]
    mz = [mz[i] for i in order]
    intensities = [intensities[i] for i in order]
    formula = [formula[i] for i in order]

    return mz, intensities, formula

def pad_mz_intensities(mz, intensities, formula, pad_length, mz_pad = 0, intensities_pad = 0, mask_missing_formula = False):

    length = len(mz)
    mz = mz + [mz_pad for _ in range(pad_length)]
    intensities = intensities + [intensities_pad for _ in range(pad_length)]
    formula = formula + ["[PAD]" for _ in range(pad_length)]
    mask = [False for _ in range(length)] + [True for _ in range(pad_length)]
    
    if mask_missing_formula: 
        mask = [m == "[PAD]" or m == "" for m in formula]
    
    assert len(mz) == len(mask)
    assert len(intensities) == len(mask)
    assert len(formula) == len(mask)

    return mz, intensities, formula, mask

def process_formula(formula, considered_atoms):

    # Split the formula on '+' signs. E.g., "C23H27ClNaO5+i" -> ["C23H27ClNaO5", "i"]
    parts = formula.split('+')
    
    # Regex to capture typical element symbols: capital letter followed by optional lowercase letters,
    # plus an optional digit count.
    element_pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
    
    element_counts = [0 for _ in considered_atoms]

    for part in parts:

        matches = element_pattern.findall(part)

        # Accumulate counts from the standard pattern
        for (element, count_str) in matches:
            count = int(count_str) if count_str else 1
            if element in considered_atoms:
                element_counts[considered_atoms.index(element)] += count
    
    return element_counts

def custom_collate_func(batch):

    # Filter out any None in the batch
    filtered_batch = [s for s in batch if s is not None]

    if len(filtered_batch) == 0:
        return None

    return torch.utils.data.dataloader.default_collate(filtered_batch)

# For proccessing of MS data 
def get_all_spectra(path):

    spectra_list = [] 
    if path.endswith(".mgf"): spectrum_generator = load_from_mgf(path, metadata_harmonization = False)
    elif path.endswith(".msp"): spectrum_generator = load_from_msp(path, metadata_harmonization = False)
    else: return []

    for _, spectrum in tqdm(enumerate(spectrum_generator)):
        spectra_list.append(spectrum)

    return spectra_list

# For processing the MS 
def bin_MS(mz, intensities, bin_resolution, max_da):

    n_bins = int(math.ceil(max_da / bin_resolution))

    mz_binned = [0 for _ in range(n_bins)]
    for m, i in zip(mz, intensities):
        
        m = math.floor(m / bin_resolution)
        if m >= n_bins: continue 
        mz_binned[m] += i

    return mz_binned

import math
import copy 
import yaml
import json
import pickle
import numpy as np
from tqdm import tqdm
from matchms.importing import load_from_mgf, load_from_msp

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

def filter_intensities(mz, intensities):
    mz_f, intensities_f = [],[]
    for m, i in zip(mz, intensities):
        if i > 0.0:
            mz_f.append(m)
            intensities_f.append(i)

    return mz_f, intensities_f

def sort_intensities(mz, intensities):

    # Filter the mz and intensities
    mz, intensities = filter_intensities(mz, intensities)

    # Sort by intensities
    order = np.argsort(intensities)[::-1]
    mz = [mz[i] for i in order]
    intensities = [intensities[i] for i  in order]

    return mz, intensities

def pad_mz_intensities(mz, intensities, pad_length, mz_pad = 0, intensities_pad = 0):

    length = len(mz)
    mz = mz + [mz_pad for _ in range(pad_length)]
    intensities = intensities + [intensities_pad for _ in range(pad_length)]
    mask = [1 for _ in range(length)] + [0 for _ in range(pad_length)]

    assert len(mz) == len(intensities)
    assert len(intensities) == len(mask)

    return mz, intensities, mask

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

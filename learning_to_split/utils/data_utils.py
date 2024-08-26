import math
import json 
import pickle
import numpy as np
from tqdm import tqdm 
from matchms.importing import load_from_mgf, load_from_msp

import torch

# For logging 
def write_dict(data_dict, path, skip = []):

    data_dict = {k: v for k, v in data_dict.items() if k not in skip}
    
    with open(path, "w") as f:
        json.dump(data_dict, f, indent = 4)

def write_json(data, path): 
    with open(path, "w", encoding = "UTF-8") as f: 
        json.dump(data, f, indent = 4)

def pickle_data(data, path):
    with open(path, "wb") as f: 
        pickle.dump(data, f)

def save_model(model, path):
    
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

# For data loading and processing 
def get_all_spectra(path):

    spectra_list = [] 
    if path.endswith(".mgf"): spectrum_generator = load_from_mgf(path, metadata_harmonization = False)
    elif path.endswith(".msp"): spectrum_generator = load_from_msp(path, metadata_harmonization = False)
    else: return []

    for _, spectrum in tqdm(enumerate(spectrum_generator)):
        spectra_list.append(spectrum)
        if _ == 100: break 

    return spectra_list 

def bin_MS(mz, intensities, max_length):

    mz_binned = [0 for _ in range(max_length)]
    for m, i in zip(mz, intensities):
        
        m = math.floor(m)
        if m >= max_length: continue 
        mz_binned[m] = i
    
    return mz_binned

# For moving the tensor to a nn tensor 
def to_tensor(tensor, to_long = False):

    # To convert to numpy array first 
    if type(tensor) is not np.array:
        tensor = np.array(tensor)

    # To convert to tensor first
    if type(tensor) is not torch.Tensor:
        tensor = torch.tensor(tensor)
        tensor = tensor.float()

    if to_long: tensor = tensor.long()        
    
    return tensor
 

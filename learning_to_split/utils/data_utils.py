import math
import json 
import numpy as np
from tqdm import tqdm 
from matchms.importing import load_from_mgf, load_from_msp

import torch

# For logging 
def write_args(args, path, skip = []):

    args_dict = args.__dict__
    args_dict = {k: v for k, v in args_dict.items() if k not in skip}
    
    with open(path, "w") as f:
        json.dump(args_dict, f, indent = 4)

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
 

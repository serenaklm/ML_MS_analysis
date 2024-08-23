import json 
import pickle
from tqdm import tqdm 
from matchms.importing import load_from_mgf, load_from_msp

# For logging 
def write_args(args, path, skip = []):

    args_dict = args.__dict__
    args_dict = {k: v for k, v in args_dict.items() if k not in skip}
    
    with open(path, "w") as f:
        json.dump(args_dict, f, indent = 4)

# For Data logging 
def get_all_spectra(path):

    spectra_list = [] 
    if path.endswith(".mgf"): spectrum_generator = load_from_mgf(path, metadata_harmonization = False)
    elif path.endswith(".msp"): spectrum_generator = load_from_msp(path, metadata_harmonization = False)
    else: return []

    for _, spectrum in tqdm(enumerate(spectrum_generator)):
        spectra_list.append(spectrum)
        
    return spectra_list 

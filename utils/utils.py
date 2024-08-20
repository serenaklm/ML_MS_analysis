import os 
import json
import wget
import zipfile
from tqdm import tqdm 

from rdkit import Chem
from matchms.importing import load_from_mgf, load_from_msp

def download_data(url, output_path):

    if os.path.exists(output_path): 
        print(f"{output_path} already exists")
    else:
        wget.download(url, out = output_path)
        print(f"downloaded {output_path}")
    
def unzip(folder, new_folder):
    with zipfile.ZipFile(folder, 'r') as zip_ref:
        zip_ref.extractall(new_folder)

def is_float(s):
    try: 
        float(s)
        return True
    except:
        return False

def get_all_spectra(path):

    spectra_list = [] 
    if path.endswith(".mgf"): spectrum_generator = load_from_mgf(path, metadata_harmonization = False)
    elif path.endswith(".msp"): spectrum_generator = load_from_msp(path, metadata_harmonization = False)
    else: return []

    for _, spectrum in tqdm(enumerate(spectrum_generator)):
        spectra_list.append(spectrum)
        
    return spectra_list

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)

    return mol

def load_json(path):
    with open(path, "r", encoding = "UTF-8") as f: 
        data = json.load(f)    
    return data 

def write_json(data, path):
    with open(path, "w", encoding = "UTF-8") as f:
        json.dump(data, f, indent = 4)
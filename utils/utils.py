import os 
import json
import wget
import zipfile

from rdkit import Chem

def download_data(url, output_path):

    if os.path.exists(output_path): 
        print(f"{output_path} already exists")
    else:
        wget.download(url, out = output_path)
        print(f"downloaded {output_path}")
    
def unzip(folder, new_folder):
    with zipfile.ZipFile(folder, 'r') as zip_ref:
        zip_ref.extractall(new_folder)

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)

    return mol
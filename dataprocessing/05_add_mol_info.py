import os
import json
import time
from utils import * 
from config import *

import pubchempy as pcp
import rdkit.Chem as Chem 
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from matchms.exporting import save_as_msp

from utils import * 
from config import *
import utils.pyclassyfire as client

def get_unique_inchikeys_smiles(MS):

    unique_inchikeys, unique_smiles = [],[]

    for r in tqdm(MS):

        inchikey = r.metadata["inchikey"]
        smiles = r.metadata["smiles"]

        if inchikey in unique_inchikeys: continue 
        unique_inchikeys.append(inchikey)
        unique_smiles.append(smiles)
    
    return unique_inchikeys, unique_smiles

def get_canonical_smiles(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    cano_smiles = Chem.CanonSmiles(smiles)

    return cano_smiles

def get_MACCS(mol):

    FP = MACCSkeys.GenMACCSKeys(mol).ToBitString()

    return FP

def get_pubchem_CACTVS(smiles):

    cid = pcp.get_cids(smiles, 'smiles')
    assert len(cid) == 1 
    cid = cid[0]
    compound = pcp.Compound.from_cid(cid)
    FP = "".join([str(bit) for bit in compound.cactvs_fingerprint])
    
    return FP

def get_morgan(mol, radius, FP_size):
    
    fpgen = AllChem.GetMorganGenerator(radius = radius, fpSize = FP_size)
    FP = fpgen.GetFingerprint(mol).ToBitString()

    return FP

def get_all_FPs(s):

    mol = Chem.MolFromSmiles(s)

    # Get various FPs
    MACCS = get_MACCS(mol)
    # pubchem_CACTVS = get_pubchem_CACTVS(smiles)

    morgan4_256 = get_morgan(mol, 2, 256)
    morgan4_1024 = get_morgan(mol, 2, 1024)
    morgan4_2048 = get_morgan(mol, 2, 2048)
    morgan4_4096 = get_morgan(mol, 2, 4096)

    morgan6_256 = get_morgan(mol, 3, 256)
    morgan6_1024 = get_morgan(mol, 3, 1024)
    morgan6_2048 = get_morgan(mol, 3, 2048)
    morgan6_4096 = get_morgan(mol, 3, 4096)

    mapping = {"MACCS": MACCS, 
                #   "CACTVS": pubchem_CACTVS,
                    "morgan4_256": morgan4_256,
                    "morgan4_1024": morgan4_1024,
                    "morgan4_2048": morgan4_2048,
                    "morgan4_4096": morgan4_4096,
                    "morgan6_256": morgan6_256,
                    "morgan6_1024": morgan6_1024,
                    "morgan6_2048": morgan6_2048,
                    "morgan6_4096": morgan6_4096}

    return mapping 

def parse_output(output):
    
    tax_fields = ('kingdom', 'superclass', 'class', 'subclass')
    values = {t: None for t in tax_fields}

    for t in tax_fields: 
        if t in output.keys():
            values[t] = output[t]["name"]
    
    return values

def get_entity(inchikey):

    output = client.get_entity(inchikey, 'json')
    output = json.loads(output)
    output = parse_output(output)

    return output

def add_info(data, mapping):

    sieved_data = [] 

    for rec in tqdm(data):
        
        if rec.metadata["inchikey"] not in mapping: continue 
        classes = mapping[rec.metadata["inchikey"]]
        classes = {k: v for k, v in classes.items() if v != "inchikey"}
        for c, v in classes.items():
            if c is None: continue 
            rec.set(c, v)
        sieved_data.append(rec)
    
    return sieved_data

def check_exists(folder, index):
    
    current_file_path = os.path.join(folder, f"{index}.json")
    if os.path.exists(current_file_path): return True 
    return False

if __name__ == "__main__":

    # # Get all the MS records
    # print("Getting all the MS now")
    # MS = get_all_spectra(os.path.join(merged_data_folder, "merged_MS.msp"))

    # Create a temp folder for this
    temp_folder = os.path.join(main_data_folder, "mol_annotations")
    if not os.path.exists(temp_folder): os.makedirs(temp_folder)
    
    # Get the mappings now
    if os.path.exists(os.path.join(temp_folder, "inchikey_and_smiles_mapping.json")):
        print("Loading the inchikey to idx mapping")
        mapping = load_json(os.path.join(temp_folder, "inchikey_and_smiles_mapping.json"))

    else:
        print("Generating the inchikey / smiles to idx mapping")

        # Get the unique inchikeys and smiles 
        unique_inchikeys, unique_smiles = get_unique_inchikeys_smiles(MS)
        mapping = {i : {"inchikey": k, "smiles": unique_smiles[i]} for i,k in enumerate(unique_inchikeys)}
        write_json(mapping, os.path.join(temp_folder, "inchikey_and_smiles_mapping.json"))

    # Iterate through to get information about each unique molecule
    for _ in range(99999):
        
        sieved_mapping = {index: key for index, key in mapping.items() if not check_exists(temp_folder, index)}

        for index, key in tqdm(sieved_mapping.items()):

            try:
                current_file_path = os.path.join(temp_folder, f"{index}.json")
                if os.path.exists(current_file_path):
                    print(f"{current_file_path} already exists. Skipping.")
                    continue
                else:
                    info = get_entity(key["inchikey"])
                    FPs = get_all_FPs(key["smiles"])
                    info.update(FPs)

                    info["inchikey"] = key["inchikey"]
                    info["smiles"] = key["smiles"]
                    write_json(info, current_file_path)
                    
                    time.sleep(15)

            except Exception as e:
                print(e)
                continue
    
    # Get the mapping now 
    entities_mapping = {} 
    for f in os.listdir(temp_folder):
        if f == "inchikey_and_smiles_mapping.json": continue
        rec = load_json(os.path.join(temp_folder, f))
        entities_mapping[rec["inchikey"]] = rec

    # Update the MS
    if not os.path.exists(final_data_folder): os.makedirs(final_data_folder)
    MS = add_info(MS, entities_mapping)
    save_as_msp(MS, os.path.join(final_data_folder, "final_data.msp"))

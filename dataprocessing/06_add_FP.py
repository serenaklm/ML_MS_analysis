import os
import json
import pubchempy as pcp
import rdkit.Chem as Chem 
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

from utils import * 
from config import *

from matchms.exporting import save_as_msp

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

def add_FPs(data):

    data_w_FPs = [] 

    for rec in tqdm(data):
        
        smiles = rec.metadata["smiles"]
        mol = Chem.MolFromSmiles(smiles)

        # Get various FPs
        MACCS = get_MACCS(mol)
        pubchem_CACTVS = get_pubchem_CACTVS(smiles)

        morgan4_256 = get_morgan(mol, 2, 256)
        morgan4_1024 = get_morgan(mol, 2, 1024)
        morgan4_2048 = get_morgan(mol, 2, 2048)
        morgan4_4096 = get_morgan(mol, 2, 4096)

        morgan6_256 = get_morgan(mol, 3, 256)
        morgan6_1024 = get_morgan(mol, 3, 1024)
        morgan6_2048 = get_morgan(mol, 3, 2048)
        morgan6_4096 = get_morgan(mol, 3, 4096)

        # Add it to the record 
        rec.set("MACCS", MACCS)
        rec.set("pubchem_CACTVS", pubchem_CACTVS)
        rec.set("morgan4_256", morgan4_256)
        rec.set("morgan4_1024", morgan4_1024)
        rec.set("morgan4_2048", morgan4_2048)
        rec.set("morgan4_4092", morgan4_4096)

        rec.set("morgan6_256", morgan6_256)
        rec.set("morgan6_1024", morgan6_1024)
        rec.set("morgan6_2048", morgan6_2048)
        rec.set("morgan6_4092", morgan6_4096)

        # Add to list of data 
        data_w_FPs.append(rec)
    
    return data_w_FPs

if __name__ == "__main__":

    # Get all the MS records 
    MS = get_all_spectra(os.path.join(main_data_folder, "data_w_classyfire_annotations", "data_w_classyfire_annotations.msp"))

    # Include various FP for each record
    if not os.path.exists(final_data_folder): os.makedirs(final_data_folder)
    MS_w_FP = add_FPs(MS)
    save_as_msp(MS_w_FP, os.path.join(final_data_folder, "final_data.msp"))
    

import os 
import numpy as np
from tqdm import tqdm 
import rdkit.Chem as Chem
from matchms import Spectrum
from matchms.importing import load_from_msp
from matchms.exporting import save_as_mgf

def check_atoms(smiles):

    allowed = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"]

    try: 
        mol = Chem.MolFromSmiles(smiles)

        for a in mol.GetAtoms():
           if a.GetSymbol() not in allowed: 
               print(a.GetSymbol())
               return False
    
    except Exception as e:
        print(e)
        return False

    return True 

if __name__ == "__main__":

    ALPHA_LIST = [0.0, 0.05, 0.10, 0.15]
    DATA_PATH = "./data/MS_final_w_subtree.msp"

    # Create the subfolder 
    for alpha in ALPHA_LIST:
        dir_name = os.path.join(f"./individual_MS_alpha_{alpha}")
        if not os.path.exists(dir_name): os.makedirs(dir_name)

    # Iterate through the MS 
    for s in tqdm(load_from_msp(DATA_PATH)):
        
        metadata = s.metadata 

        # Only look at orbitrap 
        # if metadata["instrument_type"] not in ["ESI-QFT", "APCI-QFT", "ESI-ITFT", "LC-ESI-ITFT", "LC-ESI-QFT"]: continue 

        # Filter through now 
        for alpha in ALPHA_LIST:
            
            id_ = metadata["new_id_"]
            new_mz = s.mz
            new_intensities = s.intensities 

            keep_idx = [idx for idx, iten in enumerate(new_intensities) if iten >= alpha]
            new_mz = np.array([new_mz[idx] for idx in keep_idx])
            new_intensities = np.array([new_intensities[idx] for idx in keep_idx])

            new_s = Spectrum(mz = new_mz, intensities = new_intensities, metadata = s.metadata)
            save_as_mgf(new_s, os.path.join(f"./individual_MS_alpha_{alpha}", f"{id_}.mgf"))
import os 
from tqdm import tqdm
from rdkit.Chem import AllChem
from msbuddy import assign_subformula
from msbuddy.utils import read_formula
from utils import load_pickle, scaffold_split, pickle_data, get_mol

def get_morgan(mol, radius, FP_size):
    
    fpgen = AllChem.GetMorganGenerator(radius = radius, fpSize = FP_size)
    FP = fpgen.GetFingerprint(mol).ToBitString()

    return FP

def normalize_intensity(r):

    peaks_list = r["peaks"]

    max_intensity = max([float(p["intensity"]) for p in peaks_list])
    for p in peaks_list:

        intensity_norm = float(p["intensity"]) / max_intensity * 100 
        p["mz"] = float(p["mz"])
        p["intensity_norm"] = intensity_norm

    return r

def add_info(data):

    for r in tqdm(data):

        mol = get_mol(r["smiles"])

        r["morgan4_256"] = get_morgan(mol, 2, 256)
        r["morgan4_1024"] = get_morgan(mol, 2, 1024)
        r["morgan4_2048"] = get_morgan(mol, 2, 2048)

        # # Add sub formula 
        # mz = [p["mz"] for p in r["peaks"]]
        # formula = r["formula"]
        # adduct = r["precursor_type"]

        # subformula = [f.subform_list for f in assign_subformula(mz, formula, adduct)]
        # subformula = [item[0].formula if item and hasattr(item[0], 'formula') else "" for item in subformula]

        # for i, p in enumerate(r["peaks"]):
        #     p["formula"] = read_formula(subformula[i])

if __name__ == "__main__":

    # path
    NIST_data_path = "/data/rbg/users/klingmin/projects/MS_processing/data/formatted/NIST2020_MS2.pkl"
    NIST_data = load_pickle(NIST_data_path)
    output_folder = "./data/NIST2020/MH_plus"
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # Selected ionization 
    adducts, energies, n_peaks = ["[M+H]+"], list(range(20, 30+1)), 5

    # Get a subset of these records 
    NIST_data = [normalize_intensity(r) for r in NIST_data if r["precursor_type"] in adducts and r["collision_energy"] in energies and len(r["peaks"]) >= n_peaks]
    
    # Add information to the data 
    add_info(NIST_data)

    # Perform a simple scaffold split for the data 
    train_data, val_data, test_data = scaffold_split(NIST_data)

    # Pickle the data 
    pickle_data(NIST_data, os.path.join(output_folder, "data.pkl"))
    pickle_data(train_data, os.path.join(output_folder, "train_data.pkl"))
    pickle_data(val_data, os.path.join(output_folder, "val_data.pkl"))
    pickle_data(test_data, os.path.join(output_folder, "test_data.pkl"))

    # Print the stats
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
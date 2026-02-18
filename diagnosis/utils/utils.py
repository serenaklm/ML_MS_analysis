import json
import pickle
import numpy as np
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from matchms.similarity import CosineGreedy
from matchms import calculate_scores, Spectrum

def load_pickle(path):

    with open(path, "rb") as f:

        data = pickle.load(f)
    
    return data

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def write_json(data, path):

    with open(path, "w") as f:
        json.dump(data, f, indent = 4)
    
def pickle_data(data, path):

    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_mol(smiles):

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx() + 1))
    
    return mol

# To compute the distance between 2 molecules
def compute_mol_sim(smiles1, smiles2):

    fpgen = AllChem.GetMorganGenerator(radius=2)
    
    mol1 = Chem.MolFromSmiles(smiles1)
    FP1 = fpgen.GetSparseCountFingerprint(mol1)

    mol2 = Chem.MolFromSmiles(smiles2)
    FP2 = fpgen.GetSparseCountFingerprint(mol2)

    return DataStructs.DiceSimilarity(FP1, FP2)

# To compute the distance between 2 MS
def compute_MS_sim(rec1, rec2):

    rec1_peaks = rec1["peaks"]
    rec2_peaks = rec2["peaks"]

    rec1_peaks = sorted(rec1_peaks, key = lambda x :x["mz"])
    rec2_peaks = sorted(rec2_peaks, key = lambda x :x["mz"])

    ms1 = Spectrum(mz = np.array([p["mz"] for p in rec1_peaks]),
                    intensities = np.array([p["intensity_norm"] for p in rec1_peaks]),
                    metadata = {"id": rec1["id_"],
                                "precursor_mz": rec1["precursor_MZ"]},
                    metadata_harmonization = False)

    ms2 = Spectrum(mz = np.array([p["mz"] for p in rec2_peaks]),
                    intensities = np.array([p["intensity_norm"] for p in rec2_peaks]),
                    metadata = {"id": rec2["id_"],
                                "precursor_mz": rec2["precursor_MZ"]},
                    metadata_harmonization = False)
    
    try:
        scores = calculate_scores(references=[ms1],
                                queries=[ms2],
                                similarity_function=CosineGreedy())

        sim = scores.scores.to_dict()["data"][0][0]

    except Exception as e:

        # The exception is mainly due to 0 matched peaks
        sim = 0.0

    return sim


def get_matched_CF(test, train):

    test_CF = [p["comment"]["f_pred"] for p in test["peaks"] if p["comment"]["f_pred"] != ""]
    train_CF = [p["comment"]["f_pred"] for p in train["peaks"] if p["comment"]["f_pred"] != ""]

    score = 0
    if len(test_CF) != 0: 
        matches = set(test_CF).intersection(set(train_CF))
        score = len(matches) / len(test_CF)
        
    return score

def update_dict(entry, rec, key):

    rec[key] = entry

    return rec

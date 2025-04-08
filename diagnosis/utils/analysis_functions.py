import random
import numpy as np
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from matchms.similarity import CosineGreedy
from matchms import calculate_scores, Spectrum

# To check if the energy is missing
def missing_energy(value):

    if value is None: return True
    else:
        try:
            float(value)
            return False
        except ValueError:
            return True
    
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

# To check if the two experimental conditions are identical 
def same_expt(rec1, rec2, include_energy = True):

    adduct_1 = rec1["precursor_type"]
    instrument_1 = rec1["instrument_type"]
    energy_1 = rec1["collision_energy"]
    if missing_energy(energy_1): return False

    adduct_2 = rec2["precursor_type"]
    instrument_2 = rec2["instrument_type"]
    energy_2 = rec2["collision_energy"]
    if missing_energy(energy_2): return False

    check = adduct_1 == adduct_2 and instrument_1 == instrument_2
    if include_energy:
        return check and energy_1 == energy_2
    else:
        return check  

# Consolidate a count for different experimental conditions
def get_diff_expt_param(test, train_rec, include_CE = True):

    test_adduct = test["precursor_type"]
    test_instrument = test["instrument_type"]
    test_energy = test["collision_energy"]

    diff_reasons = ["diff_adduct", "diff_instrument", "diff_CE",
                    "diff_adduct_instrument", "diff_adduct_CE",
                    "diff_instrument_CE", "diff_adduct_instrument_CE"]

    diff = {r: 0 for r in diff_reasons}

    for _, train in train_rec.items():

        train_adduct = train["precursor_type"]
        train_instrument = train["instrument_type"]
        train_energy = train["collision_energy"]
        if include_CE and not missing_energy(train_energy): continue

        check_diff_adduct = test_adduct != train_adduct
        check_diff_instrument = test_instrument != train_instrument
        check_diff_energy = test_energy != train_energy

        if check_diff_adduct and check_diff_instrument and check_diff_energy: diff["diff_adduct_instrument_CE"] += 1
        elif check_diff_instrument and check_diff_energy: diff["diff_instrument_CE"] += 1
        elif check_diff_adduct and check_diff_energy: diff["diff_adduct_CE"] += 1
        elif check_diff_adduct and check_diff_instrument: diff["diff_adduct_instrument"] += 1
        elif check_diff_energy: diff["diff_CE"] += 1
        elif check_diff_instrument: diff["diff_instrument"] += 1
        elif check_diff_adduct: diff["diff_adduct"] += 1

    # Remove some params if include_CE is False 
    if not include_CE: diff = {k: v for k,v in diff.items() if "CE" not in k}
    total = sum(diff.values())
    diff["total"] = total 
    
    return diff


# Helper function to get all the different experimental stats
def get_stats(test, train_rec, 
              sample_train = True, n_samples = 500,
              compute_mol_dist = False,
              compute_MS_dist = False):

    stats = {} 

    # Get percentage same / diff molecule 
    total_train = len(train_rec)
    n_same = len([k for k,v in train_rec.items() if v["inchikey_original"][:14] == test["inchikey_original"][:14]])
    stats["count"] = total_train
    stats["n_same_mol"] = n_same
    stats["n_diff_mol"] = total_train - n_same 

    # Get distance between molecules
    ids = list(train_rec.keys())
    if compute_mol_dist:
        if sample_train: ids = random.sample(ids, min(n_samples, len(ids)))
        mol_sim = [compute_mol_sim(test["smiles"], train_rec[id]["smiles"]) for id in ids]
        stats["mol_sim"] = sum(mol_sim)
        stats["mol_sim_n_train"] = len(ids)

    # Get distance between the MS 
    if compute_MS_dist:
        if sample_train: ids = random.sample(ids, min(n_samples, len(ids)))
        MS_sim = [compute_MS_sim(test, train_rec[id]) for id in ids]
        stats["MS_sim"] = sum(MS_sim)
        stats["MS_sim_n_train"] = len(ids)

    # Get the reason of different experimental conditions
    diff = {} 
    if not missing_energy(test["collision_energy"]): 
        diff = get_diff_expt_param(test, train_rec, include_CE = True)
    
    stats["diff_expt_count"] = diff

    return stats

    
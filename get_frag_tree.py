import os 
import time
import json
import pickle
import itertools
from tqdm import tqdm 
from pprint import pprint

import rdkit.Chem as Chem
from rdkit.Chem import Descriptors

from utils import load_pickle, get_mol, pickle_data

def get_match(mz, frags, threshold=0.01):

    matches = [] 

    for frag, rec in frags.items():
        
        bonds = rec["bonds"]
        mass = rec["mass"]

        diff_original = abs(mz - mass)
        diff_w_H = abs(mz - (mass+1.0078))
        diff_without_H = abs(mz - (mass-1.0078))
        diff_w_H2 = abs(mz - (mass+ 2 * 1.0078))
        diff_without_H2 = abs(mz - (mass- 2 * 1.0078))

        diff_w_H2O = abs(mz - (mass+18.0146))
        diff_without_H2O = abs(mz - (mass-18.0146))
        
        if diff_original < threshold: matches.append((bonds, mass, frag, diff_original, "original"))

        if diff_w_H < threshold: matches.append((bonds, mass+1.0078, frag, diff_w_H, "+ H"))
        if diff_without_H < threshold: matches.append((bonds, mass-1.0078, frag, diff_without_H, "-H"))
            
        if diff_w_H2O < threshold: matches.append((bonds, mass+18.0146, frag, diff_w_H2O, "+H2O"))
        if diff_without_H2O < threshold: matches.append((bonds, mass-18.0146, frag, diff_without_H2O, "-H2O"))

        if diff_w_H2 < threshold: matches.append((bonds, mass+2*1.0078, frag, diff_w_H2, "+ H2"))
        if diff_without_H2 < threshold: matches.append((bonds, mass-2*1.0078, frag, diff_without_H2, "-H2"))
        
    return matches

def normalize_intensity(r):

    peaks_list = r["peaks"]

    max_intensity = max([float(p["intensity"]) for p in peaks_list])
    for p in peaks_list:

        intensity_norm = float(p["intensity"]) / max_intensity * 100 
        p["mz"] = float(p["mz"])
        p["intensity_norm"] = intensity_norm

    return r

def get_fragments(mol, b):

    all_frags = {}

    frag_mol = Chem.FragmentOnBonds(mol, b)
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    
    # Calculate mass of each fragment
    for i, frag in enumerate(frags):
        frag_smiles = Chem.MolToSmiles(frag)
        if frag_smiles not in all_frags:
            mass = Descriptors.ExactMolWt(frag)
            all_frags[frag_smiles]= {"frag": frag,
                                     "bonds": b,
                                     "mass": mass}

    return all_frags

def match_mz_peaks(rec, time_limit = 30):
    
    start_time = time.time()
    
    smiles = rec["smiles"]
    mol = get_mol(smiles)
    mz = [float(p["mz"]) for p in rec["peaks"]]

    ring_bonds = [] 
    non_ring_bonds = []
    matched_mz = {z : [] for z in mz}
    skips = []

    bonds_mapping = {}

    for b in mol.GetBonds():

        if b.IsInRing(): non_ring_bonds.append([b.GetIdx()])
        else: ring_bonds.append(b.GetIdx())
        
        begin = b.GetBeginAtom().GetAtomMapNum()
        end = b.GetEndAtom().GetAtomMapNum()

        bonds_mapping[(begin,end)] = b.GetIdx()

    ring_bonds = [[c0,c1] for (c0,c1) in itertools.combinations(ring_bonds, 2)]
    all_combis = ring_bonds + non_ring_bonds

    # Start the iteration 
    i = 0 
    run = True

    while len(all_combis) > 0 and run: 

        next_combinations = [] 

        for c in all_combis:

            # Break once it is taking too long 
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                run = False 
                break 
            
            frags = get_fragments(mol, c)

            # Try to match the frags now 
            for z in mz:
                if z in skips: continue
                matches = get_match(z, frags)
                if len(matches) != 0:
                    matched_mz[z].extend(matches)
                    if z not in skips: skips.append(z)

            # Do not get next combinations once everything is found 
            if len(skips) == len(mz):
                run = False 
                break

            # Get the next combinations
            for _, v in frags.items():

                if v["mass"] < min([z for z in mz if z not in skips]): continue 
                frag = v["frag"]

                for b in frag.GetBonds():

                    begin = b.GetBeginAtom().GetAtomMapNum()
                    end = b.GetEndAtom().GetAtomMapNum()
                    if begin == 0 or end == 0: continue
                    idx = bonds_mapping[(begin, end)]
                    
                    new_c = c + [idx]
                    next_combinations.append(new_c)

        i += 1 
        if i > 3: break 

        # Update the skips 
        all_combis = next_combinations
    
    return matched_mz

def process(data):

    new_data = [] 

    for r in tqdm(data):

        frags = match_mz_peaks(r)
        peaks = r["peaks"]
        for p in peaks:
            p_frags = frags[p["mz"]]
            p["explained_peaks"] = p_frags

        new_data.append(r)

    return new_data

if __name__ == "__main__":

    # Selected ionization 
    adducts, energies, n_peaks = ["[M+H]+"], [30.0], 5

    # path
    NIST_data_path = "/data/rbg/users/klingmin/projects/MS_processing/data/formatted/NIST2020_MS2.pkl"
    NIST_data = load_pickle(NIST_data_path)

    # Get a subset of these records 
    NIST_data = [normalize_intensity(r) for r in NIST_data if r["precursor_type"] in adducts and r["collision_energy"] in energies and len(r["peaks"]) >= n_peaks]

    # Get fragments for each record
    NIST_data_w_frags = process(NIST_data)

    # Write the data 
    NIST_data_output_path = "/data/rbg/users/klingmin/projects/MS_processing/data/formatted/NIST2020_MS2_w_frags.pkl"
    pickle_data(NIST_data_w_frags, NIST_data_output_path)

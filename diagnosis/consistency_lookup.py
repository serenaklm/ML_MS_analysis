import os 
import copy
import json
import pickle
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from matchms import Spectrum
from matchms import calculate_scores
from matchms.similarity import CosineGreedy

from utils import load_pickle, pickle_data

def get_rank(scores, cand_check): 

    sorted_scores = sorted(scores, reverse=True)
    rank = 10000000

    for idx, score in enumerate(scores):
        check = cand_check[idx]
        if check: 
            rank = min(rank, sorted_scores.index(score))

    return rank

def list_to_dict_mass_specgym(data):

    data_spec = []

    for r in tqdm(data):

        metadata = copy.deepcopy(r)
        metadata["ionmode"] = "positive"
        del metadata["mzs"]
        del metadata["intensities"]
        if metadata["collision_energy"] is None: continue  
        
        mz = np.array([float(i) for i in r["mzs"].split(",")])
        intensities = np.array([float(i) for i in r["intensities"].split(",")])
        spec = Spectrum(mz = mz,
                        intensities = intensities,
                        metadata = metadata)

        data_spec.append(spec)
        # if len(data_spec) == 1000: break
        
    data_spec_dict = {r.metadata["identifier"]: r for r in data_spec}

    return data_spec, data_spec_dict


def list_to_dict_mass_NIST2023(data):

    data_spec = []

    for r in tqdm(data):

        metadata = copy.deepcopy(r)
        metadata["ionmode"] = "positive"
        metadata["identifier"] = r["spectrum_id"]
        if metadata["spectrum_type"] != "MS2": continue     
        if metadata["ionmode"] != "positive": continue    
        if metadata["collision_energy"] is None: continue  
        del metadata["peaks"]
        
        mz = np.array([float(p["mz"]) for p in r["peaks"]])
        intensities = np.array([float(p["intensity_norm"]) for p in r["peaks"]])
        spec = Spectrum(mz = mz,
                        intensities = intensities,
                        metadata = metadata)

        data_spec.append(spec)

        # if len(data_spec) == 1000: break

    data_spec_dict = {r.metadata["identifier"]: r for r in data_spec}

    return data_spec, data_spec_dict


def group_data_by_experimental_conditions(data_list):

    data_by_experimental_conditions = {} 

    for spec in tqdm(data_list): 

        formula = spec.metadata["formula"]
        instrument = spec.metadata["instrument_type"]
        adduct = spec.metadata["adduct"]
        energy = spec.metadata["collision_energy"]

        if formula not in data_by_experimental_conditions:data_by_experimental_conditions[formula] = {}
        if instrument not in data_by_experimental_conditions[formula]: data_by_experimental_conditions[formula][instrument] = {}
        if adduct not in data_by_experimental_conditions[formula][instrument]: data_by_experimental_conditions[formula][instrument][adduct] = {} 
        if energy not in data_by_experimental_conditions[formula][instrument][adduct]: data_by_experimental_conditions[formula][instrument][adduct][energy] = []

        data_by_experimental_conditions[formula][instrument][adduct][energy].append(spec)
    
    return data_by_experimental_conditions


def get_lookup_performance(data, data_by_experimental_conditions, greedy_cs):

    all_ranks = []
    clean_set, noisy_set = [], []

    for rec in tqdm(data):
            
        id_ = rec.metadata["identifier"]
        formula = rec.metadata["formula"]
        inchikey = rec.metadata["inchikey"]
        
        adduct = rec.metadata["adduct"]
        instrument = rec.metadata["instrument_type"]
        energy = rec.metadata["collision_energy"]

        if energy is None: continue

        cand_list = data_by_experimental_conditions[formula][instrument][adduct][energy]
        cand_list = [c for c in cand_list if c.metadata["identifier"] != id_]
        cand_inchikey = [c.metadata["inchikey"] for c in cand_list]
        
        # Skip if there are no candidates 
        if len(cand_list) == 0: continue

        # Skip impossible task  
        if inchikey not in cand_inchikey: continue

        # Get score for list of candidates
        try:
            scores = calculate_scores([rec], cand_list, greedy_cs).to_array()
            scores = np.vectorize(lambda x: x[0])(scores)[0]

        except Exception as e: 
            print(e)
            continue 

        # Get the ranking performance 
        cand_check = [c == inchikey for c in cand_inchikey]
        rank = get_rank(scores, cand_check)

        # Add in to the set 
        if rank == 0:
            clean_set.append(rec)
        else:
            noisy_set.append(rec)

        # Update the list of ranking results 
        all_ranks.append(rank)
    
    return all_ranks, clean_set, noisy_set


def get_recall_at_k(all_ranks):

    recall_at_k = []
    total = len(all_ranks)

    for k in range(0, 50, 1):

        recall_at_k.append(len([r for r in all_ranks if r <= k])/ total * 100)

    return recall_at_k


if __name__ == "__main__":

    # Some settings 
    greedy_cs = CosineGreedy()
    output_folder = "./results"
    cache_folder = "./cache"
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    if not os.path.exists(cache_folder): os.makedirs(cache_folder)

    # Get the paths of listed and grouped data 
    massspecgym_list_path = os.path.join(cache_folder,"massspecgym_list.pkl")
    NIST2023_list_path = os.path.join(cache_folder,"NIST2023_list.pkl")

    massspecgym_dict_path = os.path.join(cache_folder,"massspecgym_dict.pkl")
    NIST2023_dict_path = os.path.join(cache_folder,"NIST2023_dict.pkl")

    massspecgym_grouped_path = os.path.join(cache_folder,"massspecgym_grouped.pkl")
    NIST2023_grouped_path = os.path.join(cache_folder,"NIST2023_grouped.pkl")

    # Get the paths of clean & noisy set and ranking results 
    massspecgym_clean_set_path = os.path.join(output_folder,"massspecgym_clean_set.pkl")
    massspecgym_noisy_set_path = os.path.join(output_folder,"massspecgym_noisy_set.pkl")

    NIST2023_clean_set_path = os.path.join(output_folder,"NIST2023_clean_set.pkl")
    NIST2023_noisy_set_path = os.path.join(output_folder,"NIST2023_noisy_set.pkl")

    massspecgym_rankings_path = os.path.join(output_folder,"massspecgym_rankings.pkl")
    NIST2023_rankings_path = os.path.join(output_folder,"NIST2023_rankings.pkl")

    # Check if ranking results are already obtained, else proceed to get ranking results (MassSpecGym)
    if os.path.exists(massspecgym_rankings_path):

        assert os.path.exists(massspecgym_clean_set_path)
        assert os.path.exists(massspecgym_noisy_set_path)
        massspecgym_rankings = load_pickle(massspecgym_rankings_path)

    else:
        
        # 1. Check if the listed data exist; else proceed to process the data 
        if os.path.exists(massspecgym_list_path): 

            assert os.path.exists(massspecgym_dict_path)
            massspecgym_list = load_pickle(massspecgym_list_path)
        
        else: 
            
            # Load in the data 
            massspecgym_path = "/data/rbg/users/klingmin/projects/MS_processing/benchmarks/massspec_gym/MassSpecGym.tsv"
            massspecgym = pd.read_csv(massspecgym_path, sep = "\t")
            massspecgym = json.loads(massspecgym.to_json(orient = "records"))

            # Process the data 
            massspecgym_list, massspecgym_dict = list_to_dict_mass_specgym(massspecgym)

            # Pickle the data 
            pickle_data(massspecgym_list, massspecgym_list_path)
            pickle_data(massspecgym_dict, massspecgym_dict_path)

        # 2. Check if the grouped data exist; else proceed to process the data 
        if os.path.exists(massspecgym_grouped_path): 
            massspecgym_grouped = load_pickle(massspecgym_grouped_path)

        else:
            # Group data by experimental conditions 
            massspecgym_grouped = group_data_by_experimental_conditions(massspecgym_list)

            # Pickle the data 
            pickle_data(massspecgym_grouped, massspecgym_grouped_path)

        # 3. Get ranking performance 
        massspecgym_rankings, massspecgym_clean_set, massspecgym_noisy_set = get_lookup_performance(massspecgym_list, massspecgym_grouped, greedy_cs)

        pickle_data(massspecgym_rankings, os.path.join(output_folder,"massspecgym_rankings.pkl"))
        pickle_data(massspecgym_clean_set, os.path.join(output_folder,"massspecgym_clean_set.pkl"))
        pickle_data(massspecgym_noisy_set, os.path.join(output_folder,"massspecgym_noisy_set.pkl"))

    # Check if ranking results are already obtained, else proceed to get ranking results (NIST2023)
    if os.path.exists(NIST2023_rankings_path):

        assert os.path.exists(NIST2023_clean_set_path)
        assert os.path.exists(NIST2023_noisy_set_path)
        NIST2023_rankings = load_pickle(NIST2023_rankings_path)

    else:

        # 1. Check if the listed data exist; else proceed to process the data 
        if os.path.exists(NIST2023_list_path): 

            assert os.path.exists(NIST2023_dict_path)
            NIST2023_list = load_pickle(NIST2023_list_path)
        
        else: 
            
            # Load in the data 
            NIST2023_path = "/data/rbg/users/klingmin/projects/MS_processing/data/nist2023/nist2023.pkl"
            NIST2023 = load_pickle(NIST2023_path)

            # Process the data 
            NIST2023_list, NIST2023_dict = list_to_dict_mass_NIST2023(NIST2023)

            # Pickle the data 
            pickle_data(NIST2023_list, NIST2023_list_path)
            pickle_data(NIST2023_dict, NIST2023_dict_path)

        # 2. Check if the grouped data exist; else proceed to process the data 
        if os.path.exists(NIST2023_grouped_path): 
            NIST2023_grouped = load_pickle(NIST2023_grouped_path)

        else:
            # Group data by experimental conditions 
            NIST2023_grouped = group_data_by_experimental_conditions(NIST2023_list)

            # Pickle the data 
            pickle_data(NIST2023_grouped, NIST2023_grouped_path)

        # 3. Get ranking performance 
        NIST2023_rankings, NIST2023_clean_set, NIST2023_noisy_set  = get_lookup_performance(NIST2023_list, NIST2023_grouped, greedy_cs)
        pickle_data(NIST2023_rankings, os.path.join(output_folder,"NIST2023_rankings.pkl"))
        pickle_data(NIST2023_clean_set, os.path.join(output_folder,"NIST2023_clean_set.pkl"))
        pickle_data(NIST2023_noisy_set, os.path.join(output_folder,"NIST2023_noisy_set.pkl"))

    # Get the recall at k results now 
    massspecgym_recall_at_k = get_recall_at_k(massspecgym_rankings)
    NIST2023_recall_at_k = get_recall_at_k(NIST2023_rankings)

    # Get the percentage of noise in the data 
    massspecgym_percentage_noise = (len(massspecgym_rankings) - collections.Counter(massspecgym_rankings)[0]) / (len(massspecgym_rankings)) * 100
    NIST2023_percentage_noise = (len(NIST2023_rankings) - collections.Counter(NIST2023_rankings)[0]) / (len(NIST2023_rankings)) * 100

    print("Percentage noise (MassSpecGym):", massspecgym_percentage_noise)
    print("Percentage noise (NIST2023):", NIST2023_percentage_noise)

    # Plot the curve now 
    plt.figure(figsize=(8, 6))
    plt.plot(range(0, 50, 1), massspecgym_recall_at_k, label = "MassSpecGym", color = "blue")
    plt.plot(range(0, 50, 1), NIST2023_recall_at_k, label = "NIST2023", color = "green")

    # Add labels and title
    plt.xlabel('K')
    plt.ylabel('Recall@K')
    plt.title('Recall@K Curve')
    plt.legend()

    # Show the plot
    plt.show()
    plt.savefig(os.path.join(output_folder, "recall_at_k.png"))

import os 
import pickle
import matchms
import numpy as np
from tqdm import tqdm 
from pprint import pprint
from random import sample

import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from sklearn.datasets import make_regression
from sklearn.feature_selection import r_regression

from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

from matchms import Spectrum
from matchms import calculate_scores
from matchms.importing import load_from_msp
from matchms.similarity import CosineGreedy

def get_all_spectra(path):

    spectra = [] 
    i = 0 

    for s in tqdm(load_from_msp(path)):
        
        mapping = {"+": "positive",
                   "-": "negative"}
        
        s.set("ionmode", mapping[s.metadata["adduct"][-1]])
        spectra.append(s)

        i += 1
    
    return spectra

def tanimoto_similarity(list_a, list_b):
    # Ensure both lists are of equal length
    assert len(list_a) == len(list_b), "Both lists must be of equal length"
    
    # Compute the AND of the two lists (common '1's)
    intersection = sum([1 for a, b in zip(list_a, list_b) if a == 1 and b == 1])
    
    # Compute the number of '1's in each list
    sum_a = sum(list_a)
    sum_b = sum(list_b)
    
    # Calculate Tanimoto similarity
    similarity = intersection / (sum_a + sum_b - intersection)
    
    return similarity

def pairwise_tanimoto_similarity(data, key):

    n_records = len(data)
    score = np.zeros((n_records, n_records))

    for i in range(n_records):

        for j in range(i, n_records):

            if i == j: 
                score[i,j] = 1.0 
            else:
                fp1, fp2 = data[i].metadata[key], data[j].metadata[key]
                fp1 = [int(c) for c in fp1]
                fp2 = [int(c) for c in fp2]
                ts = tanimoto_similarity(fp1, fp2)
                score[i,j] = ts
                score[j,i] = ts
    
    return score

def pickle_data(data, path):
    with open(path, "wb") as f: 
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, "rb") as f: 
        return pickle.load(f)
    
if __name__ == "__main__":

    k = 10000
    data_path = "/data/rbg/users/klingmin/projects/MS_processing/data/final/MS_merged_w_mol_annotations.msp"
    deepms_model_path = "/data/rbg/users/klingmin/projects/MS_processing/models/deepms/ms2deepscore_model.pt"
    results_cache_folder = "./results_cache"
    if not os.path.exists(results_cache_folder): os.makedirs(results_cache_folder)

    data_all = get_all_spectra(data_path)
    data = sample(data_all, k)
    print(f"There are {len(data)} records.")

    greedy_cs = CosineGreedy()
    model = load_model(deepms_model_path)
    ms2deepscore = MS2DeepScore(model)


    # Greedy cosine similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_cosine_similarity.pkl")):
        pairwise_cosine_similarity_score = calculate_scores(data, data, greedy_cs, is_symmetric = True)
        pickle_data(pairwise_cosine_similarity_score, os.path.join(results_cache_folder, "pairwise_cosine_similarity.pkl"))
    else:
        pairwise_cosine_similarity_score = load_pickle(os.path.join(results_cache_folder, "pairwise_cosine_similarity.pkl"))

    # Deepms similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_deepms_score.pkl")):
        pairwise_deepms_score = calculate_scores(data, data, ms2deepscore, is_symmetric = True)
        pickle_data(pairwise_deepms_score, os.path.join(results_cache_folder, "pairwise_deepms_score.pkl"))
    else:
        pairwise_deepms_score = load_pickle(os.path.join(results_cache_folder, "pairwise_deepms_score.pkl"))

    # maccs score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_maccs_score.pkl")):
        pairwise_maccs_score = pairwise_tanimoto_similarity(data, "maccs")
        pickle_data(pairwise_maccs_score, os.path.join(results_cache_folder, "pairwise_maccs_score.pkl"))
    else:
        pairwise_maccs_score = load_pickle(os.path.join(results_cache_folder, "pairwise_maccs_score.pkl"))

    # morgan4_256 score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_morgan4_256_score.pkl")):
        pairwise_morgan4_256_score = pairwise_tanimoto_similarity(data, "morgan4_256")
        pickle_data(pairwise_morgan4_256_score, os.path.join(results_cache_folder, "pairwise_morgan4_256_score.pkl"))
    else:
        pairwise_morgan4_256_score = load_pickle(os.path.join(results_cache_folder, "pairwise_morgan4_256_score.pkl"))

    # morgan4_1024 score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_morgan4_1024_score.pkl")):
        pairwise_morgan4_1024_score = pairwise_tanimoto_similarity(data, "morgan4_1024")
        pickle_data(pairwise_morgan4_1024_score, os.path.join(results_cache_folder, "pairwise_morgan4_1024_score.pkl"))
    else:
        pairwise_morgan4_1024_score = load_pickle(os.path.join(results_cache_folder, "pairwise_morgan4_1024_score.pkl"))

    # morgan4_2048 score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_morgan4_2048_score.pkl")):
        pairwise_morgan4_2048_score = pairwise_tanimoto_similarity(data, "morgan4_2048")
        pickle_data(pairwise_morgan4_2048_score, os.path.join(results_cache_folder, "pairwise_morgan4_2048_score.pkl"))
    else:
        pairwise_morgan4_2048_score = load_pickle(os.path.join(results_cache_folder, "pairwise_morgan4_2048_score.pkl"))

    # morgan4_4096 score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_morgan4_4096_score.pkl")):
        pairwise_morgan4_4096_score = pairwise_tanimoto_similarity(data, "morgan4_4096")
        pickle_data(pairwise_morgan4_4096_score, os.path.join(results_cache_folder, "pairwise_morgan4_4096_score.pkl"))
    else:
        pairwise_morgan4_4096_score = load_pickle(os.path.join(results_cache_folder, "pairwise_morgan4_4096_score.pkl"))

    # morgan6_256 score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_morgan6_256_score.pkl")):
        pairwise_morgan6_256_score = pairwise_tanimoto_similarity(data, "morgan6_256")
        pickle_data(pairwise_morgan6_256_score, os.path.join(results_cache_folder, "pairwise_morgan6_256_score.pkl"))
    else:
        pairwise_morgan6_256_score = load_pickle(os.path.join(results_cache_folder, "pairwise_morgan6_256_score.pkl"))

    # morgan6_1024 score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_morgan6_1024_score.pkl")):
        pairwise_morgan6_1024_score = pairwise_tanimoto_similarity(data, "morgan6_1024")
        pickle_data(pairwise_morgan6_1024_score, os.path.join(results_cache_folder, "pairwise_morgan6_1024_score.pkl"))
    else:
        pairwise_morgan6_1024_score = load_pickle(os.path.join(results_cache_folder, "pairwise_morgan6_1024_score.pkl"))

    # morgan6_2048 score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_morgan6_2048_score.pkl")):
        pairwise_morgan6_2048_score = pairwise_tanimoto_similarity(data, "morgan6_2048")
        pickle_data(pairwise_morgan6_2048_score, os.path.join(results_cache_folder, "pairwise_morgan6_2048_score.pkl"))
    else:
        pairwise_morgan6_2048_score = load_pickle(os.path.join(results_cache_folder, "pairwise_morgan6_2048_score.pkl"))

    # morgan6_4096 score similarity
    if not os.path.exists(os.path.join(results_cache_folder, "pairwise_morgan6_4096_score.pkl")):
        pairwise_morgan6_4096_score = pairwise_tanimoto_similarity(data, "morgan6_4096")
        pickle_data(pairwise_morgan6_4096_score, os.path.join(results_cache_folder, "pairwise_morgan6_4096_score.pkl"))
    else:
        pairwise_morgan6_4096_score = load_pickle(os.path.join(results_cache_folder, "pairwise_morgan6_4096_score.pkl"))
        
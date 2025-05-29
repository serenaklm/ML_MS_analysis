import os 
from tqdm import tqdm
from itertools import permutations

from utils import compute_mol_sim
from utils import load_pickle, pickle_data

def get_pairwise_similarity(mol_list):

    print(f"There are {len(mol_list)} unique molecules")

    pairwise_similarity = {}

    for m_1, m_2 in tqdm(permutations(mol_list, 2)):

        sim = compute_mol_sim(m_1, m_2)
        if m_1 not in pairwise_similarity: pairwise_similarity[m_1] = {} 
        pairwise_similarity[m_1][m_2] = sim

    return pairwise_similarity

if __name__ == "__main__":

    data_folder = "/data/rbg/users/klingmin/projects/MS_processing/data/"
    datasets = ["canopus", "massspecgym", "nist2023"]

    all_mols = set() 

    for dataset in tqdm(datasets):

        data = load_pickle(os.path.join(data_folder, dataset, f"{dataset}.pkl"))
        mol_list = set([r["smiles"] for r in data])

        output_path = os.path.join(data_folder, dataset, "mol_sim.pkl")
        pairwise_similarity = get_pairwise_similarity(mol_list)
        pickle_data(pairwise_similarity, output_path)


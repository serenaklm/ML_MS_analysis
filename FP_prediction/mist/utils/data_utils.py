import os
import copy 
import yaml
import json
import pickle

import torch.nn.functional as F 

def read_config(path):

    with open(path, "r") as f:
        raw_config = yaml.load(f, Loader = yaml.Loader)
        config = copy.deepcopy(raw_config)

    return config

# For data loading and data writing 
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def pickle_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def consolidate_sampling_probability_IF(folder):

    scores = load_pickle(os.path.join(folder, "EK-FAC_scores.pkl"))["all_modules"]
    train_ids = load_pickle(os.path.join(folder, "train_ids.pkl"))

    # Get a consolidated harmful score 
    harmful_score = (scores < 0).float().mean(dim = 0).detach().cpu().numpy().tolist()

    # Format as a dictionary
    harmful_score_dict = {train_ids[i]: harmful_score[i] for i in range(len(harmful_score))}

    return harmful_score_dict
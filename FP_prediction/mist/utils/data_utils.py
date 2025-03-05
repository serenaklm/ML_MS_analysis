import copy 
import yaml
import json
import pickle

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
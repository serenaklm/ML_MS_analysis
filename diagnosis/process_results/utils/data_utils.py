import json
import pickle 

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def load_pickle(path):

    with open(path, "rb") as f:

        data = pickle.load(f)
    
    return data

def pickle_data(data, path):

    with open(path, "wb") as f:
        pickle.dump(data, f)

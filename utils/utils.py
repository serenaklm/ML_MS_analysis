import os 
import pickle 

def load_pickle(path):

    with open(path, "rb") as f:
        return pickle.load(f)
    
def pickle_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
import json 
import pickle

# For logging 
def write_args(args, path, skip = []):

    args_dict = args.__dict__
    args_dict = {k: v for k, v in args_dict.items() if k not in skip}
    
    with open(path, "w") as f:
        json.dump(args_dict, f, indent = 4)
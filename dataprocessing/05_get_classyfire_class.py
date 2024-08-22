import os
import json
from utils import * 
from config import *

import utils.pyclassyfire as client
from matchms.exporting import save_as_msp

def parse_output(output):
    
    tax_fields = ('kingdom', 'superclass', 'class', 'subclass')
    values = {t: None for t in tax_fields}

    for t in tax_fields: 
        if t in output.keys():
            values[t] = output[t]["name"]
    
    return values

def get_entity(inchikey):

    output = client.get_entity(inchikey, 'json')
    output = json.loads(output)
    output = parse_output(output)

    return output

def add_classes(data, mapping):

    sieved_data = [] 

    for rec in tqdm(data):
        
        if rec.metadata["inchikey"] not in mapping: continue 
        classes = mapping[rec.metadata["inchikey"]]
        classes = {k: v for k, v in classes.items() if v != "inchikey"}
        for c, v in classes.items():
            if c is None: continue 
            rec.set(c, v)
        sieved_data.append(rec)
    
    return sieved_data
    
if __name__ == "__main__":

    # Get all the MS records 
    MS = get_all_spectra(os.path.join(merged_data_folder, "final_MS.msp"))

    # Get the unique inchikeys 
    unique_inchikeys = list(set([s.metadata["inchikey"] for s in MS]))

    # Create a temp folder for this
    temp_folder = os.path.join(main_data_folder, "classyfire_annotations")
    if not os.path.exists(temp_folder): os.makedirs(temp_folder)

    # Get the mappings now
    if os.path.exists(os.path.join(temp_folder, "inchikey_mapping.json")):
        print("Loading the inchikey to idx mapping")
        inchikey_mapping = load_json(os.path.join(temp_folder, "inchikey_mapping.json"))

    else:
        print("Generating the inchikey to idx mapping")
        inchikey_mapping = {i : k for i,k in enumerate(unique_inchikeys)}
        write_json(inchikey_mapping, os.path.join(temp_folder, "inchikey_mapping.json"))

    # Iterate through this now 
    for index, key in tqdm(inchikey_mapping.items()):
        try:
            current_file_path = os.path.join(temp_folder, f"{index}.json")
            if os.path.exists(current_file_path): 
                continue 
            else:
                output = get_entity(key)
                output["inchikey"] = key
                write_json(output, current_file_path)
        except:
            continue
    
    # Get the mapping now 
    inchikey_entities_mapping = {} 
    for f in os.listdir(temp_folder):
        if f == "inchikey_mapping.json": continue
        rec = load_json(os.path.join(temp_folder, f))
        inchikey_entities_mapping[rec["inchikey"]] = rec

    # Update the MS
    final_w_classes_folder = os.path.join(main_data_folder, "data_w_classyfire_annotations")
    if not os.path.exists(final_w_classes_folder): os.makedirs(final_w_classes_folder)
    MS = add_classes(MS, inchikey_entities_mapping)
    save_as_msp(MS, os.path.join(final_w_classes_folder, "data_w_classyfire_annotations.msp"))
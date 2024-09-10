import os
from config import * 
from tqdm import tqdm
from utils import get_all_spectra, write_json

from matchms.exporting import save_as_msp

def assign_original_dataset(spectrum_list, key):
    
    new_spectrum_list = [] 

    for s in tqdm(spectrum_list):
        s.set("dataset", key)
        new_spectrum_list.append(s)
    
    return new_spectrum_list

def assign_new_id(spectrum_list):

    i = 0
    new_spectrum_list = []
    for s in tqdm(spectrum_list):
        s.set("new_id_", i)
        i += 1 
        new_spectrum_list.append(s)
    
    return new_spectrum_list
    
if __name__ == "__main__":

    if not os.path.exists(merged_data_folder): os.makedirs(merged_data_folder)
    n_records = {}
    all_spectrum = [] 

    for f in os.listdir(cleaned_data_folder):

        if not f.endswith(".msp"): continue
        key = f.replace(".msp", "")
        dataset = get_all_spectra(os.path.join(cleaned_data_folder, f))
        dataset = assign_original_dataset(dataset, key)
        n_records[key] = len(dataset)
        all_spectrum.extend(dataset)

    all_spectrum = assign_new_id(all_spectrum)
    print(f"There are {len(all_spectrum)} spectrum.")

    # Remove merged_MS if it already exists
    output_path = os.path.join(merged_data_folder, "merged_MS.msp")
    if os.path.exists(output_path): os.remove(output_path)

    # Write the records now
    save_as_msp(all_spectrum, output_path)
    write_json(n_records, os.path.join(merged_data_folder, "records_breakdown.json"))
import os
from config import * 
from tqdm import tqdm
from utils import get_all_spectra, write_json

from matchms.exporting import save_as_msp

def assign_original_dataset(spectrum_list, key):
    for s in tqdm(spectrum_list):
        s.set("dataset", key)

def assign_new_id(spectrum_list):

    i = 0
    for s in tqdm(spectrum_list):
        s.set("new_id_", i)
        i += 1 
    
if __name__ == "__main__":

    if not os.path.exists(merged_data_folder): os.makedirs(merged_data_folder)
    n_records = {}
    all_spectrum = [] 

    for f in os.listdir(cleaned_data_folder):

        if not f.endswith(".msp"): continue
        key = f.replace(".msp", "")
        dataset = get_all_spectra(os.path.join(cleaned_data_folder, f))
        assign_original_dataset(dataset, key)
        n_records[key] = len(dataset)
        all_spectrum.extend(dataset)

    assign_new_id(all_spectrum)
    save_as_msp(all_spectrum, os.path.join(merged_data_folder, "merged_MS.msp"))
    write_json(n_records, os.path.join(merged_data_folder, "records_breakdown.json"))
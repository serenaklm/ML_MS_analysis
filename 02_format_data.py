import os
import ast
import json
from tqdm import tqdm 

from config import * 

from matchms import Spectrum
from matchms.exporting import save_as_msp
from matchms.importing import load_from_mgf, load_from_msp

def get_all_spectra(path):

    spectra_list = [] 
    if path.endswith(".mgf"): spectrum_generator = load_from_mgf(path, metadata_harmonization = False)
    elif path.endswith(".msp"): spectrum_generator = load_from_msp(path, metadata_harmonization = False)
    else: return []

    for spectrum in tqdm(spectrum_generator):
        spectra_list.append(spectrum)
    
    return spectra_list

def process_extra_ms(data):

    processed_spectra = [] 

    for i, s in tqdm(enumerate(data)):
        
        try:
            processed_s = Spectrum(mz = s.mz,
                                intensities = s.intensities,
                                metadata = {"id": s.metadata["dataset_id"] + f"_{i}",
                                            "smiles": s.metadata["smiles"],
                                            "inchi": s.metadata["inchi"],
                                            "inchikey": s.metadata["inchiaux"],
                                            "precursor_mz": None,
                                            "adduct": s.metadata["adduct"], 
                                            "parent_mass": s.metadata["parent_mass"],
                                            "formula": s.metadata["formula"],
                                            "precursor_formula": None, 
                                            "precursor_mz": None,
                                            "instrument_type": s.metadata["instrument_type"], 
                                            "collision_energy": s.metadata["collision_energy"], 
                                            "mode": s.metadata["ionmode"],
                                            "level": s.metadata["mslevel"]})

            processed_spectra.append(processed_s)
        
        except: 
            continue
        
    return processed_spectra

def process_massbank(data):

    processed_spectra = [] 

    for s in tqdm(data):

        try:        
            processed_s = Spectrum(mz = s.mz,
                                intensities = s.intensities,
                                metadata = {"id": s.metadata["db#"],
                                            "smiles": s.metadata["smiles"],
                                            "inchi": s.metadata["inchi"],
                                            "inchikey": s.metadata["inchikey"],
                                            "precursor_mz": s.metadata["precursor_mz"],
                                            "adduct": s.metadata["adduct"], 
                                            "parent_mass": s.metadata["parent_mass"],
                                            "formula": s.metadata["formula"],
                                            "precursor_formula": None, 
                                            "precursor_mz": s.metadata["precursor_mz"],
                                            "instrument_type": s.metadata["instrument_type"], 
                                            "collision_energy": None, 
                                            "mode": s.metadata["ionmode"],
                                            "level": s.metadata["spectrum_type"]})

            processed_spectra.append(processed_s)

        except: 
            continue 

    return processed_spectra

def process_mona(data):

    processed_spectra = [] 

    for s in tqdm(data):

        try:
            processed_s = Spectrum(mz = s.mz,
                                intensities = s.intensities,
                                metadata = {"id": s.metadata["db#"],
                                            "smiles": s.metadata["smiles"],
                                            "inchi": s.metadata["inchi"],
                                            "inchikey": s.metadata["inchikey"],
                                            "precursor_mz": s.metadata["precursor_mz"],
                                            "adduct": s.metadata["adduct"], 
                                            "parent_mass": s.metadata["parent_mass"],
                                            "formula": s.metadata["formula"],
                                            "precursor_formula": None, 
                                            "precursor_mz": s.metadata["precursor_mz"],
                                            "instrument_type": s.metadata["instrument_type"], 
                                            "collision_energy": s.metadata["collision_energy"],  
                                            "mode": s.metadata["ionmode"],
                                            "level": s.metadata["spectrum_type"]})

            processed_spectra.append(processed_s)
        
        except:
            continue 

    return processed_spectra

def process_GNPS(data):

    processed_spectra = [] 

    for s in tqdm(data):

        array = s.metadata["compound_name"].split(" ")
        energy = array[-2].split(":")[-1]
        adduct = array[-1]

        try:
            processed_s = Spectrum(mz = s.mz,
                                intensities = s.intensities,
                                metadata = {"id": s.metadata["spectrum_id"],
                                            "smiles": s.metadata["smiles"],
                                            "inchi": s.metadata["inchi"],
                                            "inchikey": s.metadata["inchiaux"],
                                            "precursor_mz": None,
                                            "adduct": adduct,
                                            "parent_mass": None,
                                            "formula": None,
                                            "precursor_formula": None, 
                                            "precursor_mz": s.metadata["pepmass"][0],
                                            "instrument_type": s.metadata["source_instrument"], 
                                            "collision_energy": energy,
                                            "mode": s.metadata["ionmode"],
                                            "level": s.metadata["mslevel"]})

            processed_spectra.append(processed_s)
        
        except Exception as e:
            print(e)
            continue 

    return processed_spectra

if __name__ == "__main__":

    if not os.path.exists(processed_data_folder): os.makedirs(processed_data_folder)

    # Process the extra MS
    extra_dataset = []
    for link in extra_MS_url_list:
        file_path = os.path.join(raw_data_folder, link.split("/")[-1].split("?")[0])
        dataset = get_all_spectra(file_path)
        dataset = process_extra_ms(dataset)
        extra_dataset.extend(dataset)
    save_as_msp(extra_dataset, os.path.join(processed_data_folder, "extra_MS.msp"))

    # Process the massbank dataset
    massbank_dataset = get_all_spectra(os.path.join(raw_data_folder, "massbank_NIST.msp"))
    massbank_dataset = process_massbank(massbank_dataset)
    save_as_msp(massbank_dataset, os.path.join(processed_data_folder, "massbank.msp"))

    # Process mona
    mona_dataset = get_all_spectra(os.path.join(raw_data_folder, "mona.msp"))
    mona_dataset = process_mona(mona_dataset)
    save_as_msp(mona_dataset, os.path.join(processed_data_folder, "mona.msp"))

    # Process GNPS 
    GNPS_dataset = [] 
    for filename in GNPS_url_list:
        file_path = os.path.join(raw_data_folder, filename)
        dataset = get_all_spectra(file_path)
        dataset = process_GNPS(dataset)
        GNPS_dataset.extend(dataset)
    save_as_msp(GNPS_dataset, os.path.join(processed_data_folder, "GNPS.msp"))

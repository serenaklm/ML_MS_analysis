import os
import re
from tqdm import tqdm 

from config import * 
from utils import get_all_spectra, is_float

from matchms import Spectrum
from matchms.exporting import save_as_msp

def clean_spectrum(s):
    metadata = Spectrum.metadata_dict(s)
    metadata = {k: v for k, v in metadata.items() if v is not None}
    s_cleaned = Spectrum(mz = s.mz, intensities = s.intensities, 
                         metadata = metadata)
    return s_cleaned

def get_extra_ms_ms_level(s):
    if "mslevel" in Spectrum.metadata_dict(s): 
        return s.metadata["mslevel"]
    elif "ms_level" in Spectrum.metadata_dict(s):
        return s.metadata["ms_level"]
    else:
        return None
    
def process_extra_ms(data):

    processed_spectra = [] 

    for i, s in tqdm(enumerate(data)):
        
        try:

            ms_level = get_extra_ms_ms_level(s)
            if "collision_energy" not in s.metadata: continue

            processed_s = Spectrum(mz = s.mz,
                                   intensities = s.intensities,
                                   metadata = {"id": s.metadata["dataset_id"] + f"_{i}",
                                               "smiles": s.metadata["smiles"],
                                               "inchi": s.metadata["inchi"],
                                               "inchikey": s.metadata["inchi_aux"],
                                               "adduct": s.metadata["adduct"], 
                                               "parent_mass": s.metadata["parent_mass"],
                                               "formula": s.metadata["formula"],
                                               "pepmass": s.metadata["pepmass"],
                                               "precursor_formula": None, 
                                               "precursor_mz": None,
                                               "instrument_type": s.metadata["ion_source"] + "-" + s.metadata["instrument_type"], 
                                               "collision_energy": s.metadata["collision_energy"], 
                                               "mode": s.metadata["ionmode"],
                                               "level": ms_level})

            processed_s = clean_spectrum(processed_s)
            processed_spectra.append(processed_s)
        
        except Exception as e:
            print(e)
            continue
        
    return processed_spectra

def clean_NCE_energy(m_p, nce):

    ce = (float(m_p) * nce) / 500

    return ce 

def clean_energy_massbank(s):

    if "collision_energy" not in s.metadata:
        return None 

    collision_energy =  s.metadata["collision_energy"]

    collision_energy = collision_energy.replace("FT-MS", "")
    collision_energy = collision_energy.replace("IT-MS", "")
    collision_energy = collision_energy.replace("II", "")
    
    if "or" in collision_energy: return None 
        
    if is_float(collision_energy): 
        collision_energy = float(collision_energy)
    
    elif "ramp" in collision_energy.lower(): 
        return None 
    
    elif "ev" in collision_energy and is_float(collision_energy.replace("ev", "")): 
        collision_energy = float(collision_energy.replace("ev", ""))
        
    elif "eV" in collision_energy and is_float(collision_energy.replace("eV", "")): 
        collision_energy = float(collision_energy.replace("eV", ""))
        
    elif "V" in collision_energy and is_float(collision_energy.replace("V", "")): 
        collision_energy = float(collision_energy.replace("V", ""))
        
    elif "(nominal)" in collision_energy and "stepped" not in collision_energy:
        collision_energy = float(collision_energy.replace("(nominal)", "").replace("%", "").strip())
        collision_energy = clean_NCE_energy(s.metadata["precursor_mz"], collision_energy)
        
    elif "NCE" in collision_energy: 
        collision_energy = float(collision_energy.replace("NCE", "").replace("(", "").replace(")", ""))
        collision_energy = clean_NCE_energy(s.metadata["precursor_mz"], collision_energy)

    elif "%" in collision_energy and "stepped" not in collision_energy and "by" not in collision_energy: 
        collision_energy = float(collision_energy.replace("%", ""))
        
    else:
        return None 

    return collision_energy

def process_massbank(data):

    processed_spectra = [] 

    for _, s in tqdm(enumerate(data)):
        
        try:

            collision_energy = clean_energy_massbank(s)
            if collision_energy is None: continue

            processed_s = Spectrum(mz = s.mz,
                                intensities = s.intensities,
                                metadata = {"id": s.metadata["spectrum_id"],
                                            "smiles": s.metadata["smiles"],
                                            "inchi": s.metadata["inchi"],
                                            "inchikey": s.metadata["inchikey"],
                                            "adduct": s.metadata["adduct"], 
                                            "parent_mass": s.metadata["parent_mass"],
                                            "formula": s.metadata["formula"],
                                            "precursor_formula": None, 
                                            "precursor_mz": s.metadata["precursor_mz"],
                                            "instrument_type": s.metadata["instrument_type"], 
                                            "collision_energy": collision_energy,
                                            "mode": s.metadata["ionmode"],
                                            "level": s.metadata["ms_level"]})

            processed_s = clean_spectrum(processed_s)
            processed_spectra.append(processed_s)

        except Exception as e:
            print(e)
            continue 

    return processed_spectra

def clean_energy_mona(s):

    if "collision_energy" not in s.metadata: return None 
    if "precursor_mz" not in s.metadata: return None 

    collision_energy =  s.metadata["collision_energy"]
    precursor_mz = s.metadata["precursor_mz"]

    collision_energy = collision_energy.replace("FT-MS", "")
    collision_energy = collision_energy.replace("IT-MS", "")
    collision_energy = collision_energy.replace("II", "")
    
    if "or" in collision_energy or "->" in collision_energy: 
        return None 
        
    elif is_float(collision_energy): 
        collision_energy = float(collision_energy)
        
    elif is_float(collision_energy.replace("(", "")): 
        collision_energy = float(collision_energy.replace("(", ""))
        
    elif "HCD" in collision_energy: 
        collision_energy = collision_energy.replace("HCD", "")
        if is_float(collision_energy): 
            collision_energy = float(collision_energy)
        elif collision_energy == "(NCE 40%)": 
            collision_energy = float(collision_energy.replace("(NCE 40%)", ""))
            collision_energy = clean_NCE_energy(precursor_mz, collision_energy)

        else: return None 

    elif "ramp" in collision_energy.lower(): 
        return None  
    
    elif "ev" in collision_energy and is_float(collision_energy.replace("ev", "")): 
        collision_energy = float(collision_energy.replace("ev", ""))
        
    elif "eV" in collision_energy and is_float(collision_energy.replace("eV", "")): 
        collision_energy = float(collision_energy.replace("eV", ""))
        
    elif "V" in collision_energy and is_float(collision_energy.replace("V", "")): 
        collision_energy = float(collision_energy.replace("V", ""))

    elif "(nominal)" in collision_energy:
        collision_energy = float(collision_energy.replace("(nominal)", "").replace("%", "").strip())
        collision_energy = clean_NCE_energy(precursor_mz, collision_energy)

    elif "NCE" in collision_energy: 
        collision_energy = float(collision_energy.replace("NCE", "").replace("(", "").replace(")", ""))
        collision_energy = clean_NCE_energy(precursor_mz, collision_energy)

    elif "CE" in collision_energy: 
        collision_energy = float(collision_energy.replace("CE", ""))

    elif "%" in collision_energy: 
        collision_energy = float(collision_energy.replace("%", ""))
        
    else: 
        return None

    return collision_energy 

def process_mona(data):

    processed_spectra = [] 

    for _, s in tqdm(enumerate(data)):

        try:

            collision_energy = clean_energy_mona(s)
            if collision_energy is None: continue

            processed_s = Spectrum(mz = s.mz,
                                intensities = s.intensities,
                                metadata = {"id": s.metadata["spectrum_id"],
                                            "smiles": s.metadata["smiles"],
                                            "inchi": s.metadata["inchi"],
                                            "inchikey": s.metadata["inchikey"],
                                            "adduct": s.metadata["adduct"], 
                                            "parent_mass": s.metadata["parent_mass"],
                                            "formula": s.metadata["formula"],
                                            "precursor_formula": None, 
                                            "precursor_mz": s.metadata["precursor_mz"],
                                            "instrument_type": s.metadata["instrument_type"], 
                                            "collision_energy": collision_energy,  
                                            "mode": s.metadata["ionmode"],
                                            "level": s.metadata["ms_level"]})

            processed_s = clean_spectrum(processed_s)
            processed_spectra.append(processed_s)
        
        except Exception as e:
            print(e)
            continue 

    return processed_spectra

def process_GNPS_comment(comment, mode):
    
    try:
        # Get the collision energy (get the highest energy)
        found, add_marker = False, True
        regex_output = re.findall("CollisionEnergy:(.+) ", comment)
        if len(regex_output) == 0:
            regex_output = re.findall(r"[_-]([^-_]+)\s*eV", comment)
            add_marker = False
            found = len(regex_output) != 0
        
        else: found = True
        assert len(regex_output) == 1
        collision_energy = regex_output[0].strip().replace("\"", "")

        if add_marker:
            collision_energy = float([collision_energy[i:i+2] for i in range(0, len(collision_energy), 2)][-1])
            collision_energy = f"{collision_energy} (max)"
        else:
            collision_energy = float(collision_energy)

        # Get the adduct 
        adduct = comment.split(" ")[-1]
        if mode == "Negative": adduct = f"[{adduct}]-"
        if mode == "Positive": adduct = f"[{adduct}]+"
        
        return collision_energy, adduct
    
    except Exception as e:
        if found:
            print("Exception", e, comment)
        return None, None

def process_GNPS(data):

    processed_spectra = [] 

    for i, s in tqdm(enumerate(data)):
        
        mode = s.metadata["ionmode"]
        comment = s.metadata["compound_name"]
        collision_energy, adduct = process_GNPS_comment(comment, mode)
        if collision_energy is None: continue

        try:
            processed_s = Spectrum(mz = s.mz,
                                intensities = s.intensities,
                                metadata = {"id": s.metadata["spectrum_id"],
                                            "smiles": s.metadata["smiles"],
                                            "inchi": s.metadata["inchi"],
                                            "inchikey": s.metadata["inchi_aux"],
                                            "adduct": adduct,
                                            "parent_mass": None,
                                            "formula": None,
                                            "pepmass": s.metadata["pepmass"],
                                            "precursor_formula": None, 
                                            "precursor_mz": None,
                                            "instrument_type": s.metadata["instrument_type"], 
                                            "collision_energy": collision_energy,
                                            "mode": s.metadata["ionmode"],
                                            "level": s.metadata["ms_level"]})

            processed_s = clean_spectrum(processed_s)
            processed_spectra.append(processed_s)
        
        except Exception as e:
            print(e)
            continue 

    return processed_spectra

if __name__ == "__main__":

    if not os.path.exists(processed_data_folder): os.makedirs(processed_data_folder)

    # Process the extra MS
    if not os.path.exists(os.path.join(processed_data_folder, "extra_MS.msp")):
        extra_dataset = []
        for link in extra_MS_url_list:
            file_path = os.path.join(raw_data_folder, link.split("/")[-1].split("?")[0])
            dataset = get_all_spectra(file_path)
            dataset = process_extra_ms(dataset)
            extra_dataset.extend(dataset)

        save_as_msp(extra_dataset, os.path.join(processed_data_folder, "extra_MS.msp"))

    # Process the massbank dataset
    if not os.path.exists(os.path.join(processed_data_folder, "massbank.msp")):
        massbank_dataset = get_all_spectra(os.path.join(raw_data_folder, "massbank_NIST.msp"))
        massbank_dataset = process_massbank(massbank_dataset)
        save_as_msp(massbank_dataset, os.path.join(processed_data_folder, "massbank.msp"))

    # Process mona
    if not os.path.exists(os.path.join(processed_data_folder, "mona.msp")):
        mona_dataset = get_all_spectra(os.path.join(raw_data_folder, "mona.msp"))
        mona_dataset = process_mona(mona_dataset)
        save_as_msp(mona_dataset, os.path.join(processed_data_folder, "mona.msp"))

    # Process GNPS 
    if not os.path.exists(os.path.join(processed_data_folder, "GNPS.msp")):
        GNPS_dataset = []
        for filename in GNPS_url_list:
            file_path = os.path.join(raw_data_folder, filename)
            dataset = get_all_spectra(file_path)
            dataset = process_GNPS(dataset)
            GNPS_dataset.extend(dataset)
        save_as_msp(GNPS_dataset, os.path.join(processed_data_folder, "GNPS.msp"))

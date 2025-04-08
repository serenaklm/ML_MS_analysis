import re
import math
import copy 
import yaml
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from matchms.importing import load_from_mgf, load_from_msp

import torch 

# For data loading and data writing 
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def read_config(path):

    with open(path, "r") as f:
        raw_config = yaml.load(f, Loader = yaml.Loader)
        config = copy.deepcopy(raw_config)

    return config

# For processing of chemical formula
def process_formula(formula, considered_atoms):

    # Split the formula on '+' signs. E.g., "C23H27ClNaO5+i" -> ["C23H27ClNaO5", "i"]
    parts = formula.split('+')
    
    # Regex to capture typical element symbols: capital letter followed by optional lowercase letters,
    # plus an optional digit count.
    element_pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
    
    element_counts = [0 for _ in considered_atoms]

    for part in parts:

        matches = element_pattern.findall(part)

        # Accumulate counts from the standard pattern
        for (element, count_str) in matches:
            count = int(count_str) if count_str else 1
            if element in considered_atoms:
                element_counts[considered_atoms.index(element)] += count
    
    return element_counts

def parse_formula(formula_str):

    pattern = r"([A-Z][a-z]?)(\d*)"
    tokens = re.findall(pattern, formula_str)
    formula_dict = {}
    for (element, count_str) in tokens:
        count = int(count_str) if count_str else 1
        formula_dict[element] = formula_dict.get(element, 0) + count
    return formula_dict

def formula_dict_to_str(formula_dict):

    def hill_sort_key(element):
        if element == "C":
            return (0, element)
        elif element == "H":
            return (1, element)
        else:
            return (2, element)
    
    # Filter out elements that have zero or negative counts (just in case)
    cleaned = {el: cnt for el, cnt in formula_dict.items() if cnt > 0}
    
    # Sort elements according to the partial Hill order
    sorted_elements = sorted(cleaned.keys(), key=hill_sort_key)
    
    # Build output string
    output_str = ""
    for el in sorted_elements:
        cnt = cleaned[el]
        if cnt == 1:
            output_str += f"{el}"
        else:
            output_str += f"{el}{cnt}"
    return output_str

def parse_adduct_contents(adduct_inside):

    adduct_inside = adduct_inside.strip()
    if not adduct_inside:
        return (1, {}) 
    
    if not (adduct_inside.startswith('+') or adduct_inside.startswith('-')):
        adduct_inside = '+' + adduct_inside

    chunks = re.split(r'(?=[+-])', adduct_inside)
    chunks = [ch.strip() for ch in chunks if ch.strip()]
    
    m_multiplier = 0
    changes = {}
    
    for ch in chunks:
        sign = 1 if ch.startswith('+') else -1
        body = ch[1:].strip()

        match = re.match(r'^(\d+)?M$', body)

        if match:
            factor_str = match.group(1)
            factor = int(factor_str) if factor_str else 1
            m_multiplier += sign * factor
            continue
        
        fmatch = re.match(r'^(\d+)([A-Za-z].*)$', body)
        if fmatch:
            factor = int(fmatch.group(1))
            formula_part = fmatch.group(2)
        else:
            factor = 1
            formula_part = body
        
        sub_dict = parse_formula(formula_part)
        
        for e, c in sub_dict.items():
            changes[e] = changes.get(e, 0) + sign * factor * c

    if m_multiplier == 0:
        m_multiplier = 1
    
    return (m_multiplier, changes)

# For reading in of MS data in .mgf or .msp format
def get_all_spectra(path):

    spectra_list = [] 
    if path.endswith(".mgf"): spectrum_generator = load_from_mgf(path, metadata_harmonization = False)
    elif path.endswith(".msp"): spectrum_generator = load_from_msp(path, metadata_harmonization = False)
    else: return []

    for _, spectrum in tqdm(enumerate(spectrum_generator)):
        spectra_list.append(spectrum)

    return spectra_list

# For processing the MS 
def bin_MS(mz, intensities, bin_resolution, max_da):

    n_bins = int(math.ceil(max_da / bin_resolution))

    mz_binned = [0 for _ in range(n_bins)]
    for m, i in zip(mz, intensities):
        
        m = math.floor(m / bin_resolution)
        if m >= n_bins: continue 
        mz_binned[m] += i

    return mz_binned

def filter_intensities(mz, intensities, formula, frags):
    mz_f, intensities_f, formula_f, frags_f = [],[],[],[]
    for idx, (m, i) in enumerate(zip(mz, intensities)):
        if i > 0.0:
            mz_f.append(m)
            intensities_f.append(i)
            if formula is not None: formula_f.append(formula[idx])
            if frags is not None: frags_f.append(frags[idx])

    if len(formula_f) == 0: formula_f = None 
    if len(frags_f) == 0: frags_f = None 
    return mz_f, intensities_f, formula_f, frags_f

def sort_intensities(mz, intensities, formula, frags):

    # Filter the mz and intensities
    mz, intensities, formula, frags = filter_intensities(mz, intensities, formula, frags)

    # Sort by intensities
    order = np.argsort(intensities)[::-1]
    mz = [mz[i] for i in order]
    intensities = [intensities[i] for i in order]
    if formula is not None: formula = [formula[i] for i in order]
    
    frags_smiles, frags_weight = None, None 
    if frags is not None: 
        frags_smiles = [frags[i][0] for i in order]
        frags_weight = [frags[i][1] for i in order]

    return mz, intensities, formula, frags_smiles, frags_weight

def pad_mz_intensities(mz, intensities, formula, 
                       frags_smiles, frags_weight, 
                       pad_length, n_cands = 5, 
                       mz_pad = 0, intensities_pad = 0):

    length = len(mz)
    mz = mz + [mz_pad for _ in range(pad_length)]
    intensities = intensities + [intensities_pad for _ in range(pad_length)]
    if formula is not None: formula = formula + ["[PAD]" for _ in range(pad_length)]
    if frags_smiles is not None: frags_smiles = frags_smiles + [pad_missing_cand(n_cands) for _ in range(pad_length)]
    if frags_weight is not None: frags_weight = np.array(frags_weight + [pad_missing_cand_weight(n_cands) for _ in range(pad_length)])
    mask = [False for _ in range(length)] + [True for _ in range(pad_length)]
    
    assert len(mz) == len(mask)
    assert len(intensities) == len(mask)
    if formula is not None: assert len(formula) == len(mask)
    if frags_smiles is not None: assert len(frags_smiles) == len(mask)
    if frags_weight is not None: assert len(frags_weight) == len(mask)

    return mz, intensities, formula, frags_smiles, frags_weight, mask

# For processing of frags
def pad_missing_cand(length, pad_token = ""):
    return [pad_token for _ in range(length)]

def pad_missing_cand_weight(length):
    # Assumes uniform 
    return np.array([1.0/ length for _ in range(length)])

def replace_wildcard(smiles):

    pattern = r'\[\d+\*\]'
    return re.sub(pattern, '[*]', smiles)

def collate_candidates(candidates):

    sieved_candidates = {} 
    for c in candidates:
        smiles, loss = replace_wildcard(c[0]), c[2]
        if smiles not in sieved_candidates: sieved_candidates[smiles] = 10

        sieved_candidates[smiles] = min(sieved_candidates[smiles], loss)
    
    return sieved_candidates

def filter_candidates(frags_list, max_candidates):

    candidates = collate_candidates(frags_list)
    candidates = sorted(candidates.items(), key = lambda item: item[1])
    pad_size = max(0, max_candidates - len(candidates))

    # Filter out the top k candidates 
    candidates = candidates[:max_candidates]
    
    # Pad now 
    candidates = candidates + [("", -1e30) for _ in range(pad_size)]

    # Get candidates and their softmax values 
    smiles, weight = [c[0] for c in candidates], softmax([c[1] for c in candidates])
    
    assert len(smiles) == len(weight)
    
    return (smiles, weight)

def tokenize_frags(frags_list, tokenizer, n_cands, max_length = 128):

    # Flatten the list 
    cands = [c for f in frags_list for c in f]
    inputs = tokenizer.batch_encode_plus(cands, return_tensors = 'pt', add_special_tokens = True, 
                                         padding = 'max_length', max_length = max_length, truncation = True)
    
    # Get the tokens and mask
    tokens, mask = inputs["input_ids"], inputs["attention_mask"]

    # Reshape them now 
    tokens = tokens.view(-1, n_cands, max_length).contiguous()
    mask = mask.view(-1, n_cands, max_length).contiguous()

    return tokens, mask
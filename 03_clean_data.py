"""
    Part of code adapted from: https://github.com/pluskal-lab/MassSpecGym/blob/main/notebooks/dataset_construction/2_clean_library.ipynb 
"""

import os
import math 
import logging
from tqdm import tqdm
from typing import List

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

import matchms.filtering as msfilters
from matchms.Pipeline import Pipeline, create_workflow
from matchms.filtering.default_pipelines import DEFAULT_FILTERS, REQUIRE_COMPLETE_ANNOTATION 
from matchms.filtering.filter_utils.interpret_unknown_adduct import get_ions_from_adduct, split_ion, get_charge_of_adduct, get_multiplier_and_mass_from_adduct

from config import *
from utils import smiles_to_mol, Formula

logger = logging.getLogger("matchms")

def keep_pos_spectra(spectrum):
    if spectrum is None:
        return None
    mode = spectrum.get("mode")
    if "positive" in mode.lower():
        spectrum.set("mode", "positive")
        return spectrum
    elif mode == "P":
        spectrum.set("mode", "positive")
        return spectrum
    else:
        return None

def require_adduct_in_list(spectrum, allowed_adduct_list):

    """
        Removes spectra if the adduct is not within the given list
    """
    
    if spectrum is None: return None
    if spectrum.get("adduct") not in allowed_adduct_list:
        logger.info("Removed spectrum since adduct: %s is not in allowed_adduct_list %s", spectrum.get("adduct"), allowed_adduct_list)
        return None
    
    # Update the spectrum's adduct 
    spectrum.set("adduct", adducts_mapping[spectrum.get("adduct")])
    return spectrum

def remove_charged_molecules(spectrum):
    
    if spectrum is None: return None
    mol = smiles_to_mol(spectrum.get("smiles"))
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if charge == 0: return spectrum
    logger.info("Removed spectrum since spectrum is charged")
    return None

def require_formula_match_parent_mass(spectrum, tolerance=0.1):

    formula = spectrum.get("formula")
    if formula is None:
        logger.warning("Removed spectrum since precursor formula is None")
        return None
    formula = Formula(formula)
    if math.isclose(formula.get_mass(), float(spectrum.get("parent_mass")), abs_tol=tolerance):
        return spectrum
    else:
        logger.info(f"formula = {formula}, parent mass {spectrum.get('parent_mass')}, found mass {formula.get_mass()}")
        logger.info("mass_diff = ", float(spectrum.get("parent_mass")) - formula.get_mass())
    return None

def remove_non_ms2_spectra(spectrum):
    if spectrum.get("level") in ("MS2", "2"):
        return spectrum
    else:
        return None

def add_precursor_formula(spectrum):
    if spectrum is None:
        return None
    spectrum = spectrum.clone()
    nr_of_parent_masses, ions_split = get_ions_from_adduct(spectrum.get("adduct"))
    formula_str = spectrum.get('formula')
    if formula_str is None:
        print("No parent mass formula")
        return None
    
    original_precursor_formula = Formula(formula_str)
    new_precursor_formula = Formula("")
    for i in range(nr_of_parent_masses):
        new_precursor_formula += original_precursor_formula
    for ion in ions_split:
        sign, number, formula = split_ion(ion)
        for i in range(number):
            if sign == "+":
                new_precursor_formula += Formula(formula)
            if sign == "-":
                new_precursor_formula -= Formula(formula)
            if new_precursor_formula is None:
                return spectrum
    spectrum.set("precursor_formula", str(new_precursor_formula))
    return spectrum

def require_formula_match_parent_mass(spectrum, tolerance=0.1):
    formula = spectrum.get("formula")
    if formula is None:
        logger.warning("removed spectrum since precursor formula is None")
        return None
    formula = Formula(formula)
    if math.isclose(formula.get_mass(), float(spectrum.get("parent_mass")), abs_tol=tolerance):
        return spectrum
    else:
        logger.info(f"formula = {formula}, parent mass {spectrum.get('parent_mass')}, found mass {formula.get_mass()}")
        logger.info("mass_diff = ", float(spectrum.get("parent_mass")) - formula.get_mass())
    return None

def require_matching_adduct_precursor_mz_parent_mass(spectrum,
                                                     tolerance=0.1):
    """Checks if the adduct precursor mz and parent mass match within the tolerance"""
    if spectrum is None:
        return None

    adduct = spectrum.get("adduct")

    if adduct is None:
        logger.info("Spectrum is removed since adduct is None")
        return None
    if spectrum.get("parent_mass") is None:
        logger.info("Spectrum is removed since parent mass is None")
        return None
    if spectrum.get("precursor_mz") is None:
        logger.info("Spectrum is removed since precursor mz is None")
        return None
    try:
        precursor_mz = float(spectrum.get("precursor_mz"))
        parent_mass = float(spectrum.get("parent_mass"))
    except (TypeError, ValueError):
        logger.warning("precursor_mz or parent mass could not be converted to float, "
                       "please run add_parent_mass and add_precursor_mz first")
        return spectrum

    multiplier, correction_mass = get_multiplier_and_mass_from_adduct(adduct)
    if multiplier is None:
        logger.info("Spectrum is removed since adduct: %s could not be parsed", adduct)
        return None
    expected_parent_mass = (precursor_mz - correction_mass) / multiplier
    if not math.isclose(parent_mass, expected_parent_mass, abs_tol=tolerance):
        logger.info("Spectrum is removed because the adduct : %s and precursor_mz: %s suggest a parent mass of %s, "
                    "but parent mass %s is given",
                    adduct, precursor_mz, expected_parent_mass, parent_mass)
        return None
    return spectrum

def require_matching_adduct_and_ionmode(spectrum):
    if spectrum is None:
        return None
    ionmode = spectrum.get("ionmode")
    adduct = spectrum.get("adduct")
    charge_of_adduct = get_charge_of_adduct(adduct)
    if charge_of_adduct is None:
        return None
    if (charge_of_adduct > 0 and ionmode != "positive") or (charge_of_adduct < 0 and ionmode != "negative"):
        logger.warning("Ionmode: %s does not correspond to the charge or the adduct %s", ionmode, adduct)
        return None
    return spectrum

def derive_formula_from_smiles(spectrum_in, overwrite=True):

    def _get_formula_from_smiles(smiles):
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return CalcMolFormula(mol)

    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()
    if spectrum.get("formula") is not None:
        if overwrite is False:
            return spectrum

    formula = _get_formula_from_smiles(spectrum.get("smiles"))

    if formula is not None:
        if spectrum.get("formula") is not None:
            if spectrum.get("formula") != formula:
                logger.info("Overwriting formula from inchi. Original formula: %s New formula: %s",
                            spectrum.get('formula'), formula)
                spectrum.set("formula", formula)
        else:
            logger.info("Added formula from inchi. New Formula: %s", formula)
            spectrum.set("formula", formula)
    else:
        logger.warning("The smiles: %s could not be interpreted by rdkit, so no formula was set")
    return spectrum

def harmonize_instrument_types(spectrum, conversions: dict):
    if spectrum is None:
        return None
    spectrum = spectrum.clone()
    instrument_type = spectrum.get("instrument_type")
    if instrument_type in conversions:
        spectrum.set("instrument_type", conversions[instrument_type])
    return spectrum

def keep_energy_spectra(spectrum):

    if spectrum is None:
        return None
    energy = spectrum.get("collision_energy").strip()

    if energy in energy_mapping:
        spectrum.set("collision_energy", energy_mapping[energy])
        return spectrum
    else:
        return None

if __name__ == "__main__":

    if not os.path.exists(cleaned_data_folder): os.makedirs(cleaned_data_folder)

    workflow = create_workflow(query_filters = DEFAULT_FILTERS  + REQUIRE_COMPLETE_ANNOTATION + 
                                               [(msfilters.repair_smiles_of_salts, {"mass_tolerance": 0.1}),
                                                 msfilters.repair_not_matching_annotation, 
                                                (msfilters.require_minimum_number_of_peaks, {"n_required": 10})])

    pipeline = Pipeline(workflow)

    # Add in extra processing step 
    pipeline.processing_queries.parse_and_add_filter(remove_non_ms2_spectra, filter_position = 0)
    pipeline.processing_queries.parse_and_add_filter(keep_pos_spectra)

    pipeline.processing_queries.parse_and_add_filter((require_adduct_in_list, {"allowed_adduct_list": adducts}))
    pipeline.processing_queries.parse_and_add_filter(remove_charged_molecules)
    pipeline.processing_queries.parse_and_add_filter(require_matching_adduct_precursor_mz_parent_mass)
    pipeline.processing_queries.parse_and_add_filter(require_matching_adduct_and_ionmode)

    pipeline.processing_queries.parse_and_add_filter(derive_formula_from_smiles)
    pipeline.processing_queries.parse_and_add_filter(require_formula_match_parent_mass)
    pipeline.processing_queries.parse_and_add_filter(add_precursor_formula)

    pipeline.processing_queries.parse_and_add_filter((harmonize_instrument_types, {"conversions": instruments_mapping}))
    pipeline.processing_queries.parse_and_add_filter(keep_energy_spectra)

    for f in tqdm(os.listdir(processed_data_folder)):
        pipeline.run(os.path.join(processed_data_folder, f),
                     cleaned_query_file = os.path.join(cleaned_data_folder, f))
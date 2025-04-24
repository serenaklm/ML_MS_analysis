import numpy as np 
from typing import Any, Callable

import torch

class Data(object):

    def __init__(self, data: Any, train_mode: bool, featurizer: Callable):

        self.data = data
        self.featurizer = featurizer
        self.train_mode = train_mode

    def __getitem__(self, index: int) -> Any:

        spec, mol = self.data[index]

        mol_features = self.featurizer.featurize_mol(mol, train_mode=self.train_mode)
        spec_features = self.featurizer.featurize_spec(spec, train_mode=self.train_mode)

        # Get a dictionary of merged
        merged = {} 
        merged["spec"] = [spec_features]
        merged["mol"] = [mol_features]
        merged["spec_indices"] = [0]
        merged["mol_indices"] = [0]
        merged["matched"] = [True]

        # # Add in the mol features
        # merged["mols"] = mol_features["mols"] #[0]

        # # ['types', 'form_vec', 'ion_vec', 'intens', 'names', 'num_peaks', 'instruments', 'fingerprints', 'fingerprint_mask'])
        # # Add in the spec features
        # merged["types"] = spec_features["types"]#[0]
        # merged["form_vec"] = spec_features["form_vec"]#[0]
        # merged["ion_vec"] = spec_features["ion_vec"]#[0]
        # merged["intens"] = spec_features["intens"]#[0]
        # merged["names"] = spec_features["names"]#[0]
        # merged["num_peaks"] = spec_features["num_peaks"]#[0]
        # merged["instruments"] = spec_features["instruments"]#[0]
        # merged["fingerprints"] = spec_features["fingerprints"]#[0]
        # merged["fingerprint_mask"] = spec_features["fingerprint_mask"]#[0]

        return merged

    def __len__(self) -> int:
        return len(self.data)
    
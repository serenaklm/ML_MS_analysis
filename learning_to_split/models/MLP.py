import argparse 

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils import to_tensor
from .build import ModelFactory

@ModelFactory.register("MLP")
class MLP(nn.Module):

    def __init__(self,
                 config_dict: dict,
                 is_splitter: bool,
                 n_classes: int):
        
        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(config_dict["input_dim"], config_dict["emb_dim"]),
                                 nn.GELU(),
                                 nn.Dropout(config_dict["dropout_rate"]),
                                 nn.Linear(config_dict["emb_dim"], config_dict["hidden_dim"]),
                                 nn.GELU(),
                                 nn.Dropout(config_dict["dropout_rate"]),
                                 nn.Linear(config_dict["hidden_dim"], config_dict["hidden_dim"]),
                                 nn.GELU(),
                                 nn.Dropout(config_dict["dropout_rate"]),
                                 nn.Linear(config_dict["hidden_dim"], config_dict["output_dim"]))
        
        self.device = config_dict["device"]
        self.is_splitter = is_splitter
        if self.is_splitter: self.pred_layer = nn.Linear(config_dict["output_dim"], n_classes)

    def forward(self, mz_binned):
        
        mz_binned = to_tensor(mz_binned).to(self.device)
        mz_emb = self.MLP(mz_binned)

        return mz_emb
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

        self.MLP = nn.Sequential(nn.Linear(config_dict["input_dim"], config_dict["MLP_model_dim"]["emb_dim"]),
                                 nn.GELU(),
                                 nn.Dropout(config_dict["dropout_rate"]),
                                 nn.Linear(config_dict["MLP_model_dim"]["emb_dim"], config_dict["MLP_model_dim"]["hidden_dim"]),
                                 nn.GELU(),
                                 nn.Dropout(config_dict["dropout_rate"]),
                                 nn.Linear(config_dict["MLP_model_dim"]["hidden_dim"], config_dict["MLP_model_dim"]["hidden_dim"]),
                                 nn.GELU(),
                                 nn.Dropout(config_dict["dropout_rate"]),
                                 nn.Linear(config_dict["MLP_model_dim"]["hidden_dim"], config_dict["output_dim"]))
        
        self.device = config_dict["device"]
        self.is_splitter = is_splitter
        if self.is_splitter: 
            
            self.FP_MLP = nn.Sequential(nn.Linear(config_dict["output_dim"], config_dict["MLP_model_dim"]["emb_dim"]),
                          nn.GELU(),
                          nn.Dropout(config_dict["dropout_rate"]),
                          nn.Linear(config_dict["MLP_model_dim"]["emb_dim"], config_dict["MLP_model_dim"]["hidden_dim"]),
                          nn.GELU(),
                          nn.Dropout(config_dict["dropout_rate"]),
                          nn.Linear(config_dict["MLP_model_dim"]["hidden_dim"], config_dict["MLP_model_dim"]["hidden_dim"]),
                          nn.GELU(),
                          nn.Dropout(config_dict["dropout_rate"]),
                          nn.Linear(config_dict["MLP_model_dim"]["hidden_dim"], config_dict["output_dim"]))
            
            self.pred_layer = nn.Linear(config_dict["output_dim"], n_classes)

    def forward(self, mz_binned, FP = None):
        
        mz_binned = to_tensor(mz_binned).to(self.device)
        pred = self.MLP(mz_binned)

        if self.is_splitter: 
            pred = pred + self.FP_MLP(FP)
            pred = self.pred_layer(pred)

        return pred
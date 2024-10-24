import torch
import torch.nn as nn

from .build import ModelFactory

@ModelFactory.register("MLP")
class MLP(nn.Module):
    
    def __init__(self, is_splitter: bool = False, 
                       n_unique_adducts: int = 10, 
                       n_unique_instrument_types: int = 26,
                       input_dim: int = 888,
                       model_dim: int = 512,
                       hidden_dim: int = 2048,
                       output_dim: int = 1024,
                       dropout_rate: float = 0.2,
                       include_adduct_idx: bool = False,
                       include_instrument_idx: bool = False):
        
        super().__init__()

        # Set some params
        self.is_splitter = is_splitter 

        # Get the MLP 
        self.MLP = nn.Sequential(nn.Linear(input_dim, model_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(model_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_dim, model_dim))
        
        # Get the adduct and instrument type embeddings
        self.adduct_embedding = nn.Embedding(n_unique_adducts, model_dim)
        self.instrument_type_embedding = nn.Embedding(n_unique_instrument_types, model_dim)

        # Get the prediction layer 
        self.include_adduct_idx = include_adduct_idx
        self.include_instrument_idx = include_instrument_idx

        mul = 1
        if include_adduct_idx: mul +=1 
        if include_instrument_idx: mul +=1 
        self.pred_layer = nn.Sequential(nn.Linear(mul * model_dim, hidden_dim),
                                        nn.GELU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(hidden_dim, output_dim))

    def forward(self, batch):

        # Unpack the batch 
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]
        adduct_idx, instrument_idx = batch["adduct_idx"], batch["instrument_idx"]

        # Get the embeddings 
        binned_ms_emb = self.MLP(binned_ms)
        adduct_emb = self.adduct_embedding(adduct_idx)
        instrument_emb = self.instrument_type_embedding(instrument_idx)

        # Get the prediction 
        emb = binned_ms_emb
        if self.include_adduct_idx:
            emb = torch.concat([emb, adduct_emb], dim = -1)
        if self.include_instrument_idx:
            emb = torch.concat([emb, instrument_emb], dim = -1)

        emb = emb.contiguous()
        FP_pred = self.pred_layer(emb)

        return FP_pred
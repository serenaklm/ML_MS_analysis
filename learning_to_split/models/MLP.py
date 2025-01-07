import torch
import torch.nn as nn

from .build import ModelFactory

@ModelFactory.register("MLP")
class MLP(nn.Module):
    
    def __init__(self, is_splitter: bool = False, 
                       input_dim: int = 888,
                       model_dim: int = 512,
                       hidden_dim: int = 2048,
                       output_dim: int = 1024,
                       FP_dim: int = 256,
                       dropout_rate: float = 0.2,
                       device: torch.device = torch.device("cpu")):
        
        super().__init__()

        # Set some params
        self.is_splitter = is_splitter 
        self.device = device

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
        
        mul = 1
        if self.is_splitter: 
            mul += 1 
            self.FP_MLP = nn.Sequential(nn.Linear(FP_dim, hidden_dim),
                                              nn.GELU(),
                                              nn.Dropout(dropout_rate),
                                              nn.Linear(hidden_dim, model_dim))

        self.pred_layer = nn.Sequential(nn.Linear(mul * model_dim, hidden_dim),
                                        nn.GELU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(hidden_dim, output_dim))

    def forward(self, batch):

        # Unpack the batch 
        binned_ms = batch["binned_MS"].to(self.device)

        # Get the embeddings 
        binned_ms_emb = self.MLP(binned_ms)

        # Get the prediction 
        emb = binned_ms_emb

        # Get the FP emb if splitter 
        if self.is_splitter:
            
            FP = batch["FP"].to(self.device)
            FP_emb = self.FP_MLP(FP)
            emb = torch.concat([emb, FP_emb], dim = -1)

        emb = emb.contiguous()
        pred = self.pred_layer(emb)

        return pred
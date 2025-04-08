import torch
import torch.nn as nn

from .build import ModelFactory

@ModelFactory.register("binned_MS_encoder")

class MSBinnedModel(nn.Module):
    
    def __init__(self, is_splitter: bool = False, 
                       input_dim: int = 100,
                       model_dim: int = 512,
                       hidden_dim: int = 2048,
                       output_dim: int = 1024,
                       dropout_rate: float = 0.2,
                       FP_dim: int = 256,
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
                                 nn.Linear(hidden_dim, model_dim),
                                 nn.Dropout(dropout_rate))

        # Get the prediction layers 
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

        # Get the embeddings 
        binned_ms = batch["binned_MS"].to(self.device)
        binned_ms_emb = self.MLP(binned_ms)

        # Get the FP emb if splitter 
        if self.is_splitter:
            
            FP = batch["FP"].to(self.device)
            FP_emb = self.FP_MLP(binned_ms_emb)
            emb = torch.concat([binned_ms_emb, FP_emb], dim = -1).contiguous()

        else: 
            emb = binned_ms_emb

        # Get the predictions
        pred = self.pred_layer(emb)

        return pred
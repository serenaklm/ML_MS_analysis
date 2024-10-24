import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from .build import ModelFactory

@ModelFactory.register("MLP")
class MLP(pl.LightningModule):
    
    def __init__(self, is_splitter: bool = False, 
                       lr: float = 1e-4,
                       weight_decay: float = 0.10, 
                       n_unique_adducts: int = 10, 
                       n_unique_instrument_types: int = 26,
                       input_dim: int = 888,
                       model_dim: int = 512,
                       hidden_dim: int = 2048,
                       output_dim: int = 1024,
                       dropout_rate: float = 0.2,
                       include_adduct_idx: bool = False,
                       include_instrument_idx: bool = False,
                       config: dict = None):
        
        super().__init__()
        self.save_hyperparameters()

        # Set some params
        self.lr = lr
        self.weight_decay = weight_decay

        # Get a mean logger 
        self.avg_loss_train, self.avg_loss_val = [], []

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

    def forward(self, binned_ms, adduct_idx, instrument_idx):

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

    def compute_loss(self, FP_pred, FP):

        return F.binary_cross_entropy_with_logits(FP_pred, FP)
    
    def training_step(self, batch, batch_idx):

        # Unpack the batch 
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]
        adduct_idx, instrument_idx = batch["adduct_idx"], batch["instrument_idx"]

        # Forward pass
        FP_pred = self(binned_ms, adduct_idx, instrument_idx)

        # Compute the loss 
        loss = self.compute_loss(FP_pred, FP)
        self.log("train/loss", loss, prog_bar = True, sync_dist = True)

        # Add to the tracker 
        self.avg_loss_train.append(loss.item())

        return loss 

    def validation_step(self, batch, batch_idx):

        # Unpack the batch 
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]
        adduct_idx, instrument_idx = batch["adduct_idx"], batch["instrument_idx"]

        # Forward pass
        FP_pred = self(binned_ms, adduct_idx, instrument_idx)

        # Compute the loss 
        loss = self.compute_loss(FP_pred, FP)
        self.log("val/loss", loss, prog_bar = True, sync_dist = True)

        # Add to the tracker 
        self.avg_loss_val.append(loss.item())

    def on_validation_epoch_end(self):

        train_avg = np.mean(self.avg_loss_train)
        val_avg = np.mean(self.avg_loss_val)

        self.log("train/average_loss", train_avg, prog_bar = True, sync_dist = True)
        self.log("val/average_loss", val_avg, prog_bar = True, sync_dist = True)

        # Reset the tracker 
        self.avg_loss_train, self.avg_loss_val = [], [] 

    def configure_optimizers(self):
       
       optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)

       return optimizer
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class MSBinnedModel(pl.LightningModule):
    
    def __init__(self, lr: float = 1e-4,
                       weight_decay: float = 5e-4, 
                       input_dim: int = 100,
                       model_dim: int = 512,
                       hidden_dim: int = 2048,
                       output_dim: int = 1024,
                       pos_weight: float = 1.0,
                       reconstruction_weight: float = 1.0,
                       dropout_rate: float = 0.2,
                       include_adduct: bool = False,
                       include_CE: bool = False, 
                       include_instrument: bool = False,
                       n_adducts: int = 10, 
                       n_CEs: int = 10,
                       n_instruments: int = 10):
        
        super().__init__()
        self.save_hyperparameters()

        # Set some params
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.pos_weight = pos_weight
        self.reconstruction_weight = reconstruction_weight
        self.include_adduct = include_adduct
        self.include_CE = include_CE
        self.include_instrument = include_instrument
        
        # Get a mean logger 
        self.avg_loss_train, self.avg_loss_val = [], []

        # Get the MLP 
        feats_emb = 0
        if self.include_CE: feats_emb += model_dim
        if self.include_adduct: feats_emb += model_dim
        if self.include_instrument: feats_emb += model_dim
        self.MLP = nn.Sequential(nn.Linear(input_dim + feats_emb, model_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(model_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(hidden_dim, model_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout_rate))

        self.pred_layer = nn.Sequential(nn.Linear(model_dim, hidden_dim),
                                        nn.GELU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(hidden_dim, output_dim))
        
        # Add another loss for reconstruction 
        self.reconstruction_pred_layer = nn.Sequential(nn.Linear(model_dim, hidden_dim),
                                                       nn.GELU(),
                                                       nn.Dropout(dropout_rate),
                                                       nn.Linear(hidden_dim, input_dim))

        # Get embeddings for the adducts, CEs and instruments 
        if self.include_adduct: self.adduct_embs = nn.Embedding(n_adducts, model_dim)
        if self.include_CE: self.CE_embs = nn.Embedding(n_CEs, model_dim)
        if self.include_instrument: self.instrument_embs = nn.Embedding(n_instruments, model_dim)

    def forward(self, binned_ms, adduct, CE, instrument):

        # Get the embeddings 
        if self.include_adduct: binned_ms = torch.concat([binned_ms, self.adduct_embs(adduct)], dim = -1)
        if self.include_CE: binned_ms = torch.concat([binned_ms, self.CE_embs(CE)], dim = -1)
        if self.include_instrument: binned_ms = torch.concat([binned_ms, self.instrument_embs(instrument)], dim = -1)
        emb = self.MLP(binned_ms)

        # Get the FP prediction 
        FP_pred = self.pred_layer(emb)

        # Get the reconstruction prediction
        binned_ms_pred = self.reconstruction_pred_layer(emb)

        return FP_pred, binned_ms_pred 

    def compute_loss(self, FP_pred, FP):

        # Up weigh positive bits
        pos_weight = FP.clone().detach() * (self.pos_weight - 1) # to avoid double counting
        loss_pos = F.binary_cross_entropy_with_logits(FP_pred, FP, reduction = "none")
        loss_pos = (loss_pos * pos_weight)
        
        # Get loss for negative bits
        loss_neg = F.binary_cross_entropy_with_logits(FP_pred, FP, reduction = "none")

        # Combine the loss 
        loss = loss_pos + loss_neg
        loss = loss.mean()

        return loss
    
    def get_output(self, batch):

        # Unpack the batch 
        binned_ms = batch["binned_MS"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        FP_pred, _ = self(binned_ms, adduct, CE, instrument)

        return FP_pred
    
    def training_step(self, batch, batch_idx):

        if batch is None: return None

        # Unpack the batch 
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        FP_pred, binned_ms_pred = self(binned_ms, adduct, CE, instrument)
        print("Train", FP_pred)

        # Compute the FP prediction loss 
        FP_loss = self.compute_loss(FP_pred, FP)
    
        # Compute the reconstruction loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)

        # Get the total loss
        loss = FP_loss + self.reconstruction_weight * reconstruction_loss

        # Log the training loss
        self.log("train_FP_loss", FP_loss, prog_bar = True, sync_dist = True)
        self.log("train_reconstruction_loss", reconstruction_loss, prog_bar = True, sync_dist = True)

        # Add to the tracker 
        self.avg_loss_train.append(FP_loss.item())

        return loss 

    def validation_step(self, batch, batch_idx):

        if batch is None: return None

        # Unpack the batch 
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        FP_pred, binned_ms_pred = self(binned_ms, adduct, CE, instrument)
        print("Val", FP_pred)

        # Compute the FP prediction loss 
        FP_loss = self.compute_loss(FP_pred, FP)
        
        # Compute the reconstruction loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)

        # Log the validation loss
        self.log("val_FP_loss", FP_loss, prog_bar = True, sync_dist = True)
        self.log("val_reconstruction_loss", reconstruction_loss, prog_bar = True, sync_dist = True)

        # Add to the tracker 
        self.avg_loss_val.append(FP_loss.item())

    def on_validation_epoch_end(self):

        train_avg = np.mean(self.avg_loss_train)
        val_avg = np.mean(self.avg_loss_val)

        self.log("train_average_loss", train_avg, prog_bar = True, sync_dist = True)
        self.log("val_average_loss", val_avg, prog_bar = True, sync_dist = True)

        # Reset the tracker 
        self.avg_loss_train, self.avg_loss_val = [], [] 

    def configure_optimizers(self):
       
       optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)

       return optimizer
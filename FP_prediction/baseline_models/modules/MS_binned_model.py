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
                       dropout_rate: float = 0.2):
        
        super().__init__()
        self.save_hyperparameters()

        # Set some params
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.pos_weight = pos_weight
        self.reconstruction_weight = reconstruction_weight
        
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
                                 nn.Linear(hidden_dim, model_dim),
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

    def forward(self, binned_ms):

        # Get the embeddings 
        binned_ms_emb = self.MLP(binned_ms)

        # Get the FP prediction 
        FP_pred = self.pred_layer(binned_ms_emb)

        # Get the reconstruction prediction
        binned_ms_pred = self.reconstruction_pred_layer(binned_ms_emb)

        return FP_pred, binned_ms_pred 

    def compute_loss(self, FP_pred, FP):

        pos_weight = FP.detach().clone() * self.pos_weight

        # Let us try upweighing the positive bits 
        loss_no_reduction = F.binary_cross_entropy_with_logits(FP_pred, FP, reduction = "none")
        loss_pos = (loss_no_reduction * pos_weight).sum(-1).mean(-1)
        
        # Get the loss without upweighing the positive bits
        loss_reduced = F.binary_cross_entropy_with_logits(FP_pred, FP, reduction = "none").sum(-1).mean(-1)

        # Combine the loss 
        loss = loss_pos + loss_reduced 

        return loss
    
    def training_step(self, batch, batch_idx):

        if batch is None: return None

        # Unpack the batch 
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        # Forward pass
        FP_pred, binned_ms_pred = self(binned_ms)
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

        # Forward pass
        FP_pred, binned_ms_pred = self(binned_ms)
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
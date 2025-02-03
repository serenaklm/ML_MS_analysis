import numpy as np

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableFourierFeatures(nn.Module):
    
    def __init__(self, pos_dim, f_dim, h_dim, d_dim, gamma =1.0):  

        super(LearnableFourierFeatures, self).__init__()
        assert f_dim % 2 == 0, "f_dim must be divisible by 2."

        enc_f_dim = int(f_dim / 2)

        self.Wr = nn.Parameter(torch.randn([enc_f_dim, pos_dim]) * (gamma ** 2))
        self.MLP = nn.Sequential(nn.Linear(f_dim, h_dim),
                                 nn.GELU(),
                                 nn.Linear(h_dim, d_dim))
        
        self.div_term = np.sqrt(f_dim)

    def forward(self, x):

        # Input pos dim: (B L G M): L: sequence length. G is group = 1 for our case, M = 1 positional values
        # output dim: (B L D): D: output dim 

        XWr = torch.matmul(x, self.Wr.T)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term
        emb = self.MLP(F)

        return emb

class FragTransformerEncoder(pl.LightningModule):
    
    def __init__(self, lr: float = 1e-4,
                       weight_decay: float = 0.10, 
                       n_heads: int = 6,
                       n_layers: int = 12,
                       input_dim: int = 1000,
                       model_dim: int = 256,
                       hidden_dim: int = 4096,
                       output_dim: int = 2048,
                       pos_weight: float = 1.0,
                       reconstruction_weight: float = 1.0,
                       dropout_rate: float = 0.2,
                       chemberta_model: str = ""):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Set some params
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.pos_weight = pos_weight
        self.reconstruction_weight = reconstruction_weight

        # Get a mean logger 
        self.avg_loss_train, self.avg_loss_val = [], []

        # Get all the encoders

        
        self.intensity_encoder = LearnableFourierFeatures(1, model_dim, hidden_dim, model_dim)
        self.peaks_encoder = nn.Sequential(nn.Linear(model_dim * 2, hidden_dim),
                                           nn.GELU(),
                                           nn.Linear(hidden_dim, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = model_dim, nhead = n_heads, batch_first = True)
        self.MS_encoder = nn.TransformerEncoder(encoder_layer, num_layers = n_layers)

        self.binned_ms_encoder = nn.Sequential(nn.Linear(input_dim, model_dim),
                                               nn.GELU(),
                                               nn.Dropout(dropout_rate),
                                               nn.Linear(model_dim, hidden_dim),
                                               nn.GELU(),
                                               nn.Dropout(dropout_rate),
                                               nn.Linear(hidden_dim, hidden_dim),
                                               nn.GELU(),
                                               nn.Dropout(dropout_rate),
                                               nn.Linear(hidden_dim, model_dim))
        # Get the prediction layer 
        mul = 2
        self.pred_layer = nn.Sequential(nn.Linear(model_dim * mul, hidden_dim),
                                        nn.GELU(),
                                        nn.Linear(hidden_dim, output_dim))

        # Add another loss for reconstruction 
        self.reconstruction_pred_layer = nn.Sequential(nn.Linear(model_dim * mul, hidden_dim),
                                                       nn.GELU(),
                                                       nn.Dropout(dropout_rate),
                                                       nn.Linear(hidden_dim, input_dim))
        
    def forward(self, intensities, formula, mask, binned_ms):

        # Get binned MS emb
        binned_ms_emb = self.binned_ms_encoder(binned_ms)
        
        # Get features for the MS 
        formula_emb = self.formula_encoder(formula)
        intensities_emb = self.intensity_encoder(intensities[:, :, None])
        peaks_emb = self.peaks_encoder(torch.concat([formula_emb, intensities_emb], dim = -1))

        # Encode with transformer 
        # True are not allowed to attend while False values will be unchanged.
        MS_emb = self.MS_encoder(peaks_emb, src_key_padding_mask = mask)
        MS_emb = MS_emb[:, 0, :]

        # Merge the features together
        feats = torch.concat([MS_emb, binned_ms_emb], dim = -1)

        # Get the FP prediction 
        FP_pred = self.pred_layer(feats)

        # Get the reconstruction prediction
        binned_ms_pred = self.reconstruction_pred_layer(feats)

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

        # Unpack the batch 
        intensities, formula, mask = batch["intensities"], batch["formula"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        # Forward pass
        FP_pred, binned_ms_pred = self(intensities, formula, mask, binned_ms)

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

        # Unpack the batch 
        intensities, formula, mask = batch["intensities"], batch["formula"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        # Forward pass
        FP_pred, binned_ms_pred = self(intensities, formula, mask, binned_ms)

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
    
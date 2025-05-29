import numpy as np

import pytorch_lightning as pl

from learning_to_split import compute_gap_loss, compute_y_given_z_loss, compute_marginal_z_loss

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

class MSTransformerEncoder(pl.LightningModule):
    
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
                       include_adduct: bool = False,
                       include_CE: bool = False, 
                       include_instrument: bool = False,
                       n_adducts: int = 10, 
                       n_CEs: int = 10,
                       n_instruments: int = 10):
        
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
        self.include_adduct = include_adduct
        self.include_CE = include_CE
        self.include_instrument = include_instrument

        # Get all the encoders
        self.mz_encoder = LearnableFourierFeatures(1, model_dim, hidden_dim, model_dim)
        self.intensity_encoder = LearnableFourierFeatures(1, model_dim, hidden_dim, model_dim)
        self.peaks_encoder = nn.Sequential(nn.Linear(model_dim * 2, hidden_dim),
                                           nn.GELU(),
                                           nn.Linear(hidden_dim, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model = model_dim, nhead = n_heads, batch_first = True)
        self.MS_encoder = nn.TransformerEncoder(encoder_layer, num_layers = n_layers)
        
        feats_emb = 0 
        if self.include_CE: feats_emb += model_dim
        if self.include_adduct: feats_emb += model_dim
        if self.include_instrument: feats_emb += model_dim
        self.binned_ms_encoder = nn.Sequential(nn.Linear(input_dim + feats_emb, model_dim),
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

        # Get embeddings for the adducts, CEs and instruments 
        if self.include_adduct: self.adduct_embs = nn.Embedding(n_adducts, model_dim)
        if self.include_CE: self.CE_embs = nn.Embedding(n_CEs, model_dim)
        if self.include_instrument: self.instrument_embs = nn.Embedding(n_instruments, model_dim)

    def forward(self, mz, intensities, mask, binned_ms, adduct, CE, instrument):

        # Get the embeddings (MS_binned combined with emb for meta data)
        if self.include_adduct: binned_ms = torch.concat([binned_ms, self.adduct_embs(adduct)], dim = -1)
        if self.include_CE: binned_ms = torch.concat([binned_ms, self.CE_embs(CE)], dim = -1)
        if self.include_instrument: binned_ms = torch.concat([binned_ms, self.instrument_embs(instrument)], dim = -1)
        emb = self.binned_ms_encoder(binned_ms)

        # Get fourier features for all peaks in the MS 
        mz_emb = self.mz_encoder(mz[:, :, None])
        intensities_emb = self.intensity_encoder(intensities[:, :, None])
        peaks_emb = self.peaks_encoder(torch.concat([mz_emb, intensities_emb], dim = -1))

        # Encode with transformer 
        # True are not allowed to attend while False values will be unchanged.
        MS_emb = self.MS_encoder(peaks_emb, src_key_padding_mask = mask)
        MS_emb = MS_emb[:, 0, :]

        # Merge the features together
        feats = torch.concat([MS_emb, emb], dim = -1)

        # Get the FP prediction 
        FP_pred = self.pred_layer(feats)
        print(FP_pred)
 
        # Get the reconstruction prediction
        binned_ms_pred = self.reconstruction_pred_layer(feats)
        print(binned_ms_pred)

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
    
    def encode_spectra(self, batch):

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        pred, _ = self(mz, intensities, mask, binned_ms, adduct, CE, instrument)

        # Make sure that each bit is between 0 and 1 
        pred = F.sigmoid(pred)

        return pred
    
    def training_step(self, batch, batch_idx):

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]
        
        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]
        
        # Forward pass
        FP_pred, binned_ms_pred = self(mz, intensities, mask, binned_ms, adduct, CE, instrument)

        # Compute the FP prediction loss 
        FP_loss = self.compute_loss(FP_pred, FP)

        # Compute the reconstruction loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)

        # Get the total loss
        loss = FP_loss + self.reconstruction_weight * reconstruction_loss

        # Log the training loss
        self.log("train_FP_loss", FP_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        self.log("train_reconstruction_loss", reconstruction_loss, prog_bar = True, sync_dist = True, on_epoch = True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        FP_pred, binned_ms_pred = self(mz, intensities, mask, binned_ms, adduct, CE, instrument)

        # Compute the FP prediction loss 
        FP_loss = self.compute_loss(FP_pred, FP)

        # Compute the reconstruction loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)

        # Get the total loss
        loss = FP_loss + self.reconstruction_weight * reconstruction_loss

        # Log the validation loss
        self.log("val_FP_loss", FP_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        self.log("val_reconstruction_loss", reconstruction_loss, prog_bar = True, sync_dist = True, on_epoch = True)

        return {"loss:" : loss, "FP_loss": FP_loss, "reconstruction_loss": reconstruction_loss}

    def test_step(self, batch, batch_idx):

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Forward pass
        FP_pred, binned_ms_pred = self(mz, intensities, mask, binned_ms, adduct, CE, instrument)

        # Compute the FP prediction loss 
        FP_loss = self.compute_loss(FP_pred, FP)

        # Compute the reconstruction loss
        reconstruction_loss = F.mse_loss(binned_ms_pred, binned_ms)

        # Get the total loss
        loss = FP_loss + self.reconstruction_weight * reconstruction_loss

        # Log the test loss
        self.log("test_FP_loss", FP_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        self.log("test_reconstruction_loss", reconstruction_loss, prog_bar = True, sync_dist = True, on_epoch = True)

        return {"loss:" : loss, "FP_loss": FP_loss, "reconstruction_loss": reconstruction_loss}

    def configure_optimizers(self):
       
       optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)

       return optimizer

class MSTransformerEncoderSplitter(pl.LightningModule):

    def __init__(self, lr: float = 1e-4,
                       weight_decay: float = 0.10, 
                       w_gap: float = 1.0,
                       w_ratio: float = 1.0,
                       w_balance: float = 1.0,
                       tar_ratio: float = 0.8,
                       n_heads: int = 6,
                       n_layers: int = 12,
                       input_dim: int = 1000,
                       model_dim: int = 256,
                       hidden_dim: int = 4096,
                       FP_dim: int = 4096,
                       dropout_rate: float = 0.2,
                       include_adduct: bool = False,
                       include_CE: bool = False, 
                       include_instrument: bool = False,
                       n_adducts: int = 10, 
                       n_CEs: int = 10,
                       n_instruments: int = 10,
                       jaccard_threshold: float = 0.75):

        super().__init__()
  
        # Set some params
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.include_adduct = include_adduct
        self.include_CE = include_CE
        self.include_instrument = include_instrument

        # Get the training params 
        self.w_gap = w_gap 
        self.w_ratio = w_ratio 
        self.w_balance = w_balance 
        self.total = w_gap + w_ratio #+ w_balance
        self.tar_ratio = tar_ratio
        self.jaccard_threshold = jaccard_threshold
        self.lr = lr
        self.weight_decay = weight_decay

        # Get all the encoders
        self.mz_encoder = LearnableFourierFeatures(1, model_dim, hidden_dim, model_dim)
        self.intensity_encoder = LearnableFourierFeatures(1, model_dim, hidden_dim, model_dim)
        self.peaks_encoder = nn.Sequential(nn.Linear(model_dim * 2, hidden_dim),
                                           nn.GELU(),
                                           nn.Linear(hidden_dim, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model = model_dim, nhead = n_heads, batch_first = True)
        self.MS_encoder = nn.TransformerEncoder(encoder_layer, num_layers = n_layers)
        
        feats_emb = 0 
        if self.include_CE: feats_emb += model_dim
        if self.include_adduct: feats_emb += model_dim
        if self.include_instrument: feats_emb += model_dim
        self.binned_ms_encoder = nn.Sequential(nn.Linear(input_dim + feats_emb, model_dim),
                                               nn.GELU(),
                                               nn.Dropout(dropout_rate),
                                               nn.Linear(model_dim, hidden_dim),
                                               nn.GELU(),
                                               nn.Dropout(dropout_rate),
                                               nn.Linear(hidden_dim, hidden_dim),
                                               nn.GELU(),
                                               nn.Dropout(dropout_rate),
                                               nn.Linear(hidden_dim, model_dim))
        # Get the FP encoder 
        self.FP_encoder = nn.Sequential(nn.Linear(FP_dim, hidden_dim),
                                       nn.GELU(),
                                       nn.Linear(hidden_dim, model_dim),
                                       nn.Dropout(dropout_rate))
        
        # Get the merger layer 
        # Merge binned_ms_emb, MS_emb, FP_emb
        self.merge_layer = nn.Sequential(nn.Linear(model_dim * 3, hidden_dim),
                                         nn.GELU(),
                                         nn.Linear(hidden_dim, model_dim))
        
        # Get the prediction layer 
        self.pred_layer = nn.Sequential(nn.Linear(model_dim, hidden_dim),
                                        nn.GELU(),
                                        nn.Linear(hidden_dim, 2),
                                        nn.Softmax(dim = -1))

        # Get embeddings for the adducts, CEs and instruments 
        if self.include_adduct: self.adduct_embs = nn.Embedding(n_adducts, model_dim)
        if self.include_CE: self.CE_embs = nn.Embedding(n_CEs, model_dim)
        if self.include_instrument: self.instrument_embs = nn.Embedding(n_instruments, model_dim)

    def add_predictor(self, predictor):
        self.predictor = predictor
        self.predictor.eval()

    @torch.no_grad()
    def _get_FP(self, batch):

        FP_pred = self.predictor.encode_spectra(batch)

        return FP_pred

    def encode_spectra(self, batch):

        # Get the inputs and move to the device
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]
        FP = batch["FP"]

        adduct, CE, instrument = None, None, None
        if self.include_adduct: adduct = batch["adduct"]
        if self.include_CE: CE = batch["CE"]
        if self.include_instrument: instrument = batch["instrument"]

        # Get the embeddings (MS_binned combined with emb for meta data)
        if self.include_adduct: binned_ms = torch.concat([binned_ms, self.adduct_embs(adduct)], dim = -1)
        if self.include_CE: binned_ms = torch.concat([binned_ms, self.CE_embs(CE)], dim = -1)
        if self.include_instrument: binned_ms = torch.concat([binned_ms, self.instrument_embs(instrument)], dim = -1)
        emb = self.binned_ms_encoder(binned_ms)

        # Get fourier features for all peaks in the MS 
        mz_emb = self.mz_encoder(mz[:, :, None])
        intensities_emb = self.intensity_encoder(intensities[:, :, None])
        peaks_emb = self.peaks_encoder(torch.concat([mz_emb, intensities_emb], dim = -1))

        # Encode with transformer 
        # True are not allowed to attend while False values will be unchanged.
        MS_emb = self.MS_encoder(peaks_emb, src_key_padding_mask = mask)
        MS_emb = MS_emb[:, 0, :]

        # Add in the info of FP 
        FP_emb = self.FP_encoder(FP)
        feats = self.merge_layer(torch.concat([MS_emb, emb, FP_emb], dim = -1))

        # Get the FP prediction 
        pred = self.pred_layer(feats)

        return pred

    def training_step(self, batch, batch_idx):
        
        """Training step"""

        pred = self.encode_spectra(batch)
        FP = batch["FP"]
        
        # Get the gap loss 
        FP_pred = self._get_FP(batch)
        gap_loss = compute_gap_loss(pred, FP_pred, FP, jaccard_threshold= self.jaccard_threshold)

        # Get the balance loss
        # balance_loss = compute_y_given_z_loss(pred, FP)

        # Get the ratio loss 
        ratio_loss, _ = compute_marginal_z_loss(pred, tar_ratio = self.tar_ratio)

        # Get the total loss 
        loss = (self.w_gap * gap_loss + self.w_ratio * ratio_loss) / self.total 

        self.log("splitter/gap_loss", gap_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        # self.log("splitter/balance_loss", balance_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        self.log("splitter/ratio_loss", ratio_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        self.log("splitter/loss", loss, prog_bar = True, sync_dist = True, on_epoch = True)

        return {"gap_loss": gap_loss, "ratio_loss": ratio_loss, "loss": loss}

    def get_output(self, batch):

        pred = self.encode_spectra(batch)
        return pred
    
    def configure_optimizers(self):
       
       optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)

       return optimizer
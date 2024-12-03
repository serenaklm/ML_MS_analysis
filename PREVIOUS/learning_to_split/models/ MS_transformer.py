import torch
import torch.nn as nn

from .build import ModelFactory

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
    
@ModelFactory.register("MSTransformer")
class MSTransformer(nn.Module):
    
    def __init__(self, is_splitter: bool = False,
                       n_heads: int = 6,
                       n_layers: int = 12,
                       input_dim: int = 1000,
                       model_dim: int = 256,
                       hidden_dim: int = 4096,
                       output_dim: int = 2048,
                       FP_dim: int = 256,
                       n_unique_adducts: int = 10,
                       n_unique_instrument_types: int = 30,
                       include_adduct_idx: bool = False,
                       include_instrument_idx: bool = False,
                       dropout_rate: float = 0.2):
        
        super().__init__()

        # Set some params
        self.is_splitter = is_splitter 
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.mz_encoder = LearnableFourierFeatures(1, model_dim, hidden_dim, model_dim)
        self.intensity_encoder = LearnableFourierFeatures(1, model_dim, hidden_dim, model_dim)
        self.peaks_encoder = nn.Sequential(nn.Linear(model_dim * 2, hidden_dim),
                                           nn.GELU(),
                                           nn.Linear(hidden_dim, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model = model_dim, nhead = n_heads, batch_first = True)
        self.MS_encoder = nn.TransformerEncoder(encoder_layer, num_layers = n_layers)

        self.adduct_embedding = nn.Embedding(n_unique_adducts, model_dim)
        self.instrument_type_embedding = nn.Embedding(n_unique_instrument_types, model_dim)

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
        self.include_adduct_idx = include_adduct_idx
        self.include_instrument_idx = include_instrument_idx

        mul = 2
        if include_adduct_idx: mul +=1 
        if include_instrument_idx: mul +=1 

        self.pred_layer = nn.Sequential(nn.Linear(model_dim * mul, hidden_dim),
                                        nn.GELU(),
                                        nn.Linear(hidden_dim, output_dim))

        if self.is_splitter: 
            mul += 1 
            self.FP_MLP = nn.Sequential(nn.Linear(FP_dim, hidden_dim),
                                              nn.GELU(),
                                              nn.Dropout(dropout_rate),
                                              nn.Linear(hidden_dim, model_dim))

    def forward(self, batch):

        # Unpack the batch 
        mz, intensities, mask = batch["mz"], batch["intensities"], batch["mask"]
        binned_ms = batch["binned_MS"]
        adduct_idx, instrument_idx = batch["adduct_idx"], batch["instrument_idx"]
        FP = batch["FP"]

        # Get binned MS emb
        binned_ms_emb = self.binned_ms_encoder(binned_ms)
        
        # Get fourier features for the MS 
        mz_emb = self.mz_encoder(mz[:, :, None])
        intensities_emb = self.intensity_encoder(intensities[:, :, None])
        peaks_emb = self.peaks_encoder(torch.concat([mz_emb, intensities_emb], dim = -1))

        # Encode with transformer 
        # True are not allowed to attend while False values will be unchanged.
        MS_emb = self.MS_encoder(peaks_emb, src_key_padding_mask = mask)
        MS_emb = MS_emb[:, 0, :]

        # Get adduct and instrument emb
        adduct_emb = self.adduct_embedding(adduct_idx)
        instrument_type_emb = self.instrument_type_embedding(instrument_idx)

        # Merge the features together
        emb = torch.concat([MS_emb, binned_ms_emb], dim = -1)
        if self.include_adduct_idx:
            emb = torch.concat([emb, adduct_emb], dim = -1)

        if self.include_adduct_idx:
            emb = torch.concat([emb, instrument_type_emb], dim = -1)

        # Get the FP emb if splitter 
        if self.is_splitter:
            
            FP = batch["FP"]
            FP_emb = self.FP_MLP(FP)
            emb = torch.concat([emb, FP_emb], dim = -1)

        pred = self.MLP(emb)
        
        return pred
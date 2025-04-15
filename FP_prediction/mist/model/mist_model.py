""" mist_model.py
"""
import math 
from functools import partial
from typing import Optional, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from modules import modules
from mist.data import featurizers

from learning_to_split import compute_marginal_z_loss, compute_y_given_z_loss, compute_gap_loss

def cosine_loss(x, y):

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    return 1 - cosine_sim(
        x.expand(y.shape), y.float()
    ).unsqueeze(-1)

def BCE_loss(FP_pred, FP, pos_weight = 5):

    # Up weigh positive bits
    pos_weight = FP.clone().detach() * (pos_weight - 1) # to avoid double counting
    loss_pos = F.binary_cross_entropy(FP_pred, FP, reduction = "none")
    loss_pos = (loss_pos * pos_weight)
    
    # Get loss for negative bits
    loss_neg = F.binary_cross_entropy(FP_pred, FP, reduction = "none")

    # Combine the loss 
    loss = loss_pos + loss_neg

    return loss

def build_lr_scheduler(
    optimizer, lr_decay_frac: float, decay_steps: int = 10000, warmup: int = 100
):
    """_summary_

    Args:
        optimizer (_type_): _description_
        lr_decay_frac (float): _description_
        decay_steps (int, optional): _description_. Defaults to 10000.
        warmup (int, optional): _description_. Defaults to 100.
    """

    def lr_lambda(step):
        if step >= warmup:
            # Adjust
            step = step - warmup
            rate = lr_decay_frac ** (step // decay_steps)
        else:
            rate = 1 - math.exp(-step / warmup)
        return rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

class MistNet(pl.LightningModule):

    def __init__(
        self,
        magma_aux_loss: bool = False,
        form_embedder: str = "float",
        magma_loss_lambda: int = 0,
        iterative_preds: str = "none",
        iterative_loss_weight: float = 0.0,
        shuffle_train: bool = False,
        fp_names: List[str] = ["morgan2048"],
        binarization_thresh: float = 0.5,
        loss_fn: str = "bce",
        pos_weight: int = 5, 
        
        hidden_size: int = 128,
        peak_attn_layers: int = 2,
        set_pooling: str = "inten",
        spectra_dropout: float = 0.1,
        num_heads: int = 8,
        pairwise_featurization: bool = True,
        embed_instrument: bool = False,
        no_diffs: bool = False,

        refine_layers: int = 4,
        magma_modulo: int = 512,

        learning_rate: float = 0.1,
        lr_decay_frac: float = 0.1,
        weight_decay: float = 0.0,
        scheduler: bool = False
    ):
        """_summary_

        Args:
            fp_names (List[str], optional): _description_. Defaults to ["morgan2048"].
            binarization_thresh (float, optional): _description_. Defaults to 0.5.
            loss_fn (str, optional): _description_. Defaults to "bce".
            magma_aux_loss (bool, optional): _description_. Defaults to False.
            iterative_preds (str, optional): _description_. Defaults to "none".
            iterative_loss_weight (float, optional): _description_. Defaults to 0.0.
            shuffle_train (bool, optional): _description_. Defaults to False.
            magma_loss_lambda (int, optional): _description_. Defaults to 0.
            form_embedder (str, optional): _description_. Defaults to "float".

        Raises:
            NotImplementedError: _description_
        """

        super().__init__()
        self.save_hyperparameters()
        self.predict_frag_fps = magma_aux_loss
        self.form_embedder = form_embedder
        self.magma_loss_lambda = magma_loss_lambda

        self.iterative_preds = iterative_preds
        self.iterative_loss_weight = iterative_loss_weight
        self.shuffle_train = shuffle_train
        self.output_size = featurizers.FingerprintFeaturizer.get_fingerprint_size(
            fp_names
        )
        # Bit thresh
        self.thresh = binarization_thresh

        # BCE loss
        self.bce_loss = partial(BCE_loss, pos_weight = pos_weight)
        self.loss_name = loss_fn
        self.cosine_loss = cosine_loss

        if self.loss_name == "bce":
            self.loss_fn = self.bce_loss
        elif self.loss_name == "mse":
            mse_loss = nn.MSELoss(reduction="none")
            self.loss_fn = mse_loss
        elif self.loss_name == "cosine":
            self.loss_fn = self.cosine_loss
        else:
            raise NotImplementedError()

        self.spectra_encoder = self._build_model(hidden_size = hidden_size, 
                                                 peak_attn_layers = peak_attn_layers,
                                                 set_pooling = set_pooling,
                                                 spectra_dropout = spectra_dropout, 
                                                 pairwise_featurization = pairwise_featurization,
                                                 num_heads = num_heads,
                                                 embed_instrument = embed_instrument,
                                                 no_diffs = no_diffs,
                                                 refine_layers = refine_layers,
                                                 magma_modulo=magma_modulo)

        # Useful for shuffle_train
        rand_perm = torch.randperm(self.output_size).long()
        inv_perm = torch.argsort(rand_perm)

        self.rand_ordering = nn.Parameter(rand_perm, requires_grad=False)
        self.inv_ordering = nn.Parameter(inv_perm, requires_grad=False)

        # For optmizer 
        self.learning_rate = learning_rate
        self.lr_decay_frac = lr_decay_frac
        self.weight_decay = weight_decay 
        self.scheduler = scheduler

    def _build_model(
        self,
        hidden_size: int = 50,
        peak_attn_layers: int = 2,
        set_pooling: str = "cls",
        spectra_dropout: float = 0.0,
        pairwise_featurization: bool = True,
        num_heads: int = 8,
        embed_instrument: bool = False,
        no_diffs: bool = True,
        top_layers: int = 1,
        refine_layers: int = 0,
        magma_modulo: int = 2048,
    ):
        
        """build_model"""

        self.hidden_size = hidden_size
        self.magma_modulo = magma_modulo

        # Can only be less than or equal to 2048
        # if self.magma_modulo > 2048:
        #    raise ValueError()
        spectra_encoder_main = modules.FormulaTransformer(
            hidden_size=hidden_size,
            peak_attn_layers=peak_attn_layers,
            set_pooling=set_pooling,
            spectra_dropout=spectra_dropout,
            pairwise_featurization=pairwise_featurization,
            num_heads=num_heads,
            form_embedder=self.form_embedder,
            embed_instrument=embed_instrument,
            no_diffs=no_diffs,
        )
        
        fragment_pred_parts = []
        for _ in range(top_layers - 1):
            fragment_pred_parts.append(nn.Linear(hidden_size, hidden_size))
            fragment_pred_parts.append(nn.ReLU())
            fragment_pred_parts.append(nn.Dropout(spectra_dropout))

        fragment_pred_parts.append(nn.Linear(hidden_size, magma_modulo))
        fragment_predictor = nn.Sequential(*fragment_pred_parts)

        if self.iterative_preds == "none":
            top_layer_parts = []
            for _ in range(top_layers - 1):
                top_layer_parts.append(nn.Linear(hidden_size, hidden_size))
                top_layer_parts.append(nn.ReLU())
                top_layer_parts.append(nn.Dropout(spectra_dropout))

            top_layer_parts.append(nn.Linear(hidden_size, self.output_size))
            top_layer_parts.append(nn.Sigmoid())
            spectra_predictor = nn.Sequential(*top_layer_parts)
        elif self.iterative_preds in ["growing"]:
            spectra_predictor = modules.FPGrowingModule(
                hidden_input_dim=hidden_size,
                final_target_dim=self.output_size,
                num_splits=refine_layers,
                reduce_factor=2,
            )
        else:
            raise NotImplementedError()

        module_list = [spectra_encoder_main, fragment_predictor, spectra_predictor]
        spectra_encoder = nn.ModuleList(module_list)
        return spectra_encoder

    def encode_mol(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """encode_mol.

        Identity encoder because we want to predict fingerprints

        """
        return batch["mols"][:, :], {}

    def _calc_frag_fps_loss(self, target_frag_fps, pred_frag_fps, frag_fps_mask):
        """_calc_frag_fps_loss.

        Args:
            target_frag_fps:
            pred_frag_fps:
            frag_fps_mask:
        """

        # Compute fp shape
        pred_shape = pred_frag_fps.shape[-1]
        targ_shape = target_frag_fps.shape[-1]
        reshaped_preds = pred_frag_fps.reshape(-1, pred_shape)
        reshaped_targs = target_frag_fps.reshape(-1, targ_shape)

        # Fold fingerprint if needed on the fly
        if pred_shape < targ_shape:
            # Find bits where
            batch_ind, bit_ind = torch.where(reshaped_targs)
            bit_ind = bit_ind % pred_shape

            # Develop new target
            reshaped_targs = torch.zeros_like(reshaped_preds.detach())
            reshaped_targs[batch_ind, bit_ind] += 1

            # Clamp s.t. this is equal to 1
            reshaped_targs = torch.clamp(reshaped_targs, max=1)

        # target_frag_fps shape: B x Np x fingerprint size
        # Transpose pred_frag_fps to be same shape as target_frag_fps
        # pred_frag_fps = pred_frag_fps.transpose(0, 1)
        reshaped_preds = torch.sigmoid(reshaped_preds)
        frag_fps_loss = self.loss_fn(reshaped_preds, reshaped_targs)
        frag_fps_loss = frag_fps_loss.mean(-1)
        frag_fps_loss = frag_fps_loss.reshape(*frag_fps_mask.shape)
        frag_fps_loss = frag_fps_loss * frag_fps_mask[:, :]  # , None]

        # Mean over cross entropy, sum over last dim
        frag_fps_loss = frag_fps_loss.sum(-1)

        # Divide by num entries in each row for a row-wise mean
        num_entries = frag_fps_mask.sum(-1)
        frag_fps_loss = frag_fps_loss / (num_entries + 1e-12)
        frag_fps_loss[num_entries == 0] = 0

        # Verify there's something to predict
        if num_entries.sum() > 0:
            return frag_fps_loss
        else:
            return torch.zeros_like(frag_fps_loss)

    def compute_loss(
        self,
        pred_fp,
        target_fp,
        fingerprints=None,
        fingerprint_mask=None,
        aux_outputs_mol={},
        aux_outputs_spec={},
        train_step=True,
        **kwargs,
    ):
        """compute_loss."""
        # Compute weight of loss function
        ret_dict = {}
        fp_loss, magma_loss, iterative_loss = None, None, None

        # Get FP Loss
        print(pred_fp)
        fp_loss_full = self.loss_fn(pred_fp, target_fp)
        fp_loss = fp_loss_full.mean(-1)
        pred_frag_fps = aux_outputs_spec.get("pred_frag_fps", None)

        ret_dict["fp_loss"] = fp_loss.mean().item()

        # Magma
        if self.predict_frag_fps:
            frag_fps_loss_mean = self._calc_frag_fps_loss(
                fingerprints, pred_frag_fps, fingerprint_mask
            )
            weighted_frag_fps_loss = (
                torch.tensor(self.magma_loss_lambda, device=self.device)
                * frag_fps_loss_mean
            )

            magma_loss = weighted_frag_fps_loss
            ret_dict["frag_loss"] = magma_loss.mean().item()

        # Iterative predictions
        if self.iterative_preds == "none":
            pass
        elif self.iterative_preds in ["growing"]:

            cur_targ = target_fp
            if self.shuffle_train:
                cur_targ = cur_targ[:, self.rand_ordering]

            int_preds = aux_outputs_spec["int_preds"][::-1]
            aux_loss = None
            for int_pred in int_preds:

                targ_shape = int_pred.shape[-1]

                # Find bits where
                batch_ind, bit_ind = torch.where(cur_targ)
                bit_ind = bit_ind % targ_shape

                # Develop new target
                new_targ = torch.zeros_like(int_pred.detach())
                new_targ[batch_ind, bit_ind] += 1

                # Clamp s.t. this is equal to 1
                new_targ = torch.clamp(new_targ, max=1)

                # Compute loss
                temp_loss = self.loss_fn(int_pred, new_targ).mean(-1)
                ret_dict[f"loss_on_{targ_shape}"] = temp_loss.mean().item()
                if aux_loss is None:
                    aux_loss = temp_loss
                else:
                    aux_loss += temp_loss
                cur_targ = new_targ

            if aux_loss is not None:
                ret_dict["aux_iterative_loss"] = aux_loss.mean().item()
                iterative_loss = self.iterative_loss_weight * aux_loss

        # Pull losses together
        # Add new fp loss
        total_loss = torch.clone(fp_loss)
        loss_weights = torch.ones_like(total_loss)
        loss_weights = loss_weights / loss_weights.sum()
        if magma_loss is not None:
            total_loss += magma_loss
        if iterative_loss is not None:
            total_loss += iterative_loss

        # Weighted mean over batch
        total_loss = (total_loss * loss_weights).sum()
        ret_dict["loss"] = total_loss.mean()
        ret_dict["mol_loss"] = fp_loss.mean()

        if self.predict_frag_fps:

            ret_dict["frag_loss"] = magma_loss.mean()

        # Overwrite for non train step
        if not train_step:
            ret_dict["loss"] = ret_dict["mol_loss"]
            ret_dict["cos_loss"] = self.cosine_loss(pred_fp, target_fp).mean()
            ret_dict["bce_loss"] = self.bce_loss(pred_fp, target_fp).mean()

        return ret_dict

    def training_step(self, batch, batch_idx):
        """training_step.

        This is called by lightning trainer.

        Returns loss obj

        """
        # Sum pool over channels for simplicity
        pred_fp, aux_outputs_spec = self.encode_spectra(batch)

        # Mol fp's
        target_fp_all, aux_outputs_mol = self.encode_mol(batch)
        target_fp_all = target_fp_all.float()

        # Get the proper matched indices
        mol_inds = batch["mol_indices"]
        norm_inds = mol_inds[batch["matched"]]
        target_fp = target_fp_all[norm_inds]

        # Get the magma fingerprints 
        magma_fingerprints = batch.get("fingerprints")
        if magma_fingerprints is not None:
            magma_fingerprints[magma_fingerprints == -1] = 0
        
        # Compute loss and log
        ret_dict = self.compute_loss(
            pred_fp,
            target_fp,
            aux_outputs_mol=aux_outputs_mol,
            aux_outputs_spec=aux_outputs_spec,
            fingerprints=magma_fingerprints,
            fingerprint_mask=batch.get("fingerprint_mask"),
            train_step=True,
        )
        for k, v in ret_dict.items():
            self.log(
                f"train_{k}",
                v,
                batch_size=len(pred_fp),
                on_epoch=True,
                logger=True,
            )
        return ret_dict

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        pred_fp, aux_outputs_spec = self.encode_spectra(batch)

        # Mol fp's
        target_fp_all, aux_outputs_mol = self.encode_mol(batch)
        target_fp_all = target_fp_all.float()

        # Get the proper matched indices
        mol_inds = batch["mol_indices"]
        norm_inds = mol_inds[batch["matched"]]
        target_fp = target_fp_all[norm_inds]

        # Get the magma fingerprints 
        magma_fingerprints = batch.get("fingerprints")
        if magma_fingerprints is not None:
            magma_fingerprints[magma_fingerprints == -1] = 0

        # Compute loss and log
        ret_dict = self.compute_loss(
            pred_fp,
            target_fp,
            aux_outputs_mol=aux_outputs_mol,
            aux_outputs_spec=aux_outputs_spec,
            fingerprints=magma_fingerprints,
            fingerprint_mask=batch.get("fingerprint_mask"),
            train_step=False,
        )
        for k, v in ret_dict.items():
            self.log(f"val_{k}", v, batch_size=len(pred_fp), logger=True, on_epoch=True)
        return ret_dict

    def test_step(self, batch, batch_idx):
        """Test step"""
        pred_fp, aux_outputs = self.encode_spectra(batch)

        # Mol fp's
        target_fp = self.encode_mol(batch)[0].float()

        if "mol_indices" in batch:
            mol_inds = batch["mol_indices"]
            norm_inds = mol_inds[batch["matched"]]
            target_fp = target_fp[norm_inds]

        mol_bce_loss = self.loss_fn(pred_fp, target_fp)
        mol_bce_loss_mean = mol_bce_loss.mean()

        self.log(
            "test_loss",
            mol_bce_loss_mean,
            batch_size=len(pred_fp),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": mol_bce_loss_mean}

    def encode_spectra(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        
        """encode_spectra."""
        if self.iterative_preds == "none":
            encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)

            pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
            aux_outputs = {"pred_frag_fps": pred_frag_fps}

            output = self.spectra_encoder[2](encoder_output)
            aux_outputs["h0"] = encoder_output

        elif self.iterative_preds == "growing":
            encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)
            pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
            aux_outputs = {"pred_frag_fps": pred_frag_fps}

            output = self.spectra_encoder[2](encoder_output)
            intermediates = output[:-1]
            final_output = output[-1]
            aux_outputs["int_preds"] = intermediates
            output = final_output
            aux_outputs["h0"] = encoder_output
        else:
            raise NotImplementedError()
        return output, aux_outputs

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        ## Scheduler
        if not self.scheduler:
            return optimizer
        else:
            scheduler = build_lr_scheduler(
                optimizer,
                lr_decay_frac=self.lr_decay_frac,
                decay_steps=self.lr_decay_time,
                warmup=100,
            )
            ret = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": "step",
                },
            }
            return ret

class MistNetSplitter(pl.LightningModule):

    def __init__(self,
                 form_embedder: str = "float",
                 binarization_thresh: float = 0.5,
                 tar_ratio: float = 0.8,
                 hidden_size: int = 128,
                 peak_attn_layers: int = 2,
                 set_pooling: str = "inten",
                 spectra_dropout: float = 0.1,
                 num_heads: int = 8,
                 pairwise_featurization: bool = True,
                 embed_instrument: bool = False,
                 no_diffs: bool = False,

                 w_gap: float = 1.0, 
                 w_ratio: float = 1.0, 
                 w_balance: float = 1.0,

                 learning_rate: float = 0.1,
                 lr_decay_frac: float = 0.1,
                 weight_decay: float = 0.0,
                 scheduler: bool = False):
        
        super().__init__()
        self.form_embedder = form_embedder

        # Bit thresh
        self.tar_ratio = tar_ratio
        self.w_gap = w_gap
        self.w_ratio = w_ratio
        self.w_balance = w_balance
        self.total = w_gap + w_ratio + w_balance

        self.thresh = binarization_thresh

        self.spectra_encoder = self._build_model(hidden_size = hidden_size, 
                                                 peak_attn_layers = peak_attn_layers,
                                                 set_pooling = set_pooling,
                                                 spectra_dropout = spectra_dropout, 
                                                 pairwise_featurization = pairwise_featurization,
                                                 num_heads = num_heads,
                                                 embed_instrument = embed_instrument,
                                                 no_diffs = no_diffs)

        # For optmizer 
        self.learning_rate = learning_rate
        self.lr_decay_frac = lr_decay_frac
        self.weight_decay = weight_decay 
        self.scheduler = scheduler
    
    def _build_model(
        self,
        hidden_size: int = 50,
        peak_attn_layers: int = 2,
        set_pooling: str = "cls",
        spectra_dropout: float = 0.0,
        pairwise_featurization: bool = True,
        num_heads: int = 8,
        embed_instrument: bool = False,
        no_diffs: bool = True
    ):
        
        """build_model"""

        self.hidden_size = hidden_size
        spectra_encoder_main = modules.FormulaTransformer(
            hidden_size=hidden_size,
            peak_attn_layers=peak_attn_layers,
            set_pooling=set_pooling,
            spectra_dropout=spectra_dropout,
            pairwise_featurization=pairwise_featurization,
            num_heads=num_heads,
            form_embedder=self.form_embedder,
            embed_instrument=embed_instrument,
            no_diffs=no_diffs,
        )

        top_layer_parts = []
        top_layer_parts.append(nn.Linear(hidden_size, hidden_size))
        top_layer_parts.append(nn.ReLU())
        top_layer_parts.append(nn.Dropout(spectra_dropout))

        top_layer_parts.append(nn.Linear(hidden_size, 2))
        spectra_predictor = nn.Sequential(*top_layer_parts)

        module_list = [spectra_encoder_main, spectra_predictor]
        spectra_encoder = nn.ModuleList(module_list)

        return spectra_encoder

    def encode_spectra(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        
        """encode_spectra."""

        encoder_output, _ = self.spectra_encoder[0](batch, return_aux=True)
        output = self.spectra_encoder[1](encoder_output)

        return output

    def add_predictor(self, predictor):
        self.predictor = predictor
        self.predictor.eval()

    @torch.no_grad()
    def _get_FP(self, batch):

        FP_pred, _ = self.predictor.encode_spectra(batch)
        return FP_pred

    def training_step(self, batch, batch_idx):
        
        """Training step"""
        pred = self.encode_spectra(batch)
        FP = batch["mols"]
        
        # Get the gap loss 
        FP_pred = self._get_FP(batch)
        gap_loss = compute_gap_loss(pred, FP_pred, FP, threshold = self.thresh)

        # Get the balance loss
        balance_loss = compute_y_given_z_loss(pred, FP)

        # Get the ratio loss 
        ratio_loss, _ = compute_marginal_z_loss(pred, tar_ratio = self.tar_ratio)

        # Get the total loss 
        loss = (self.w_gap * gap_loss + self.w_balance * balance_loss + self.w_ratio * ratio_loss) / self.total 

        self.log("splitter/gap_loss", gap_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        self.log("splitter/balance_loss", balance_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        self.log("splitter/ratio_loss", ratio_loss, prog_bar = True, sync_dist = True, on_epoch = True)
        self.log("splitter/loss", loss, prog_bar = True, sync_dist = True, on_epoch = True)

        return {"gap_loss": gap_loss, "balance_loss": balance_loss, "ratio_loss": ratio_loss, "loss": loss}

    def get_output(self, batch):
        
        pred = self.encode_spectra(batch)
        return pred 

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        ## Scheduler
        if not self.scheduler:
            return optimizer
        else:
            scheduler = build_lr_scheduler(
                optimizer,
                lr_decay_frac=self.lr_decay_frac,
                decay_steps=self.lr_decay_time,
                warmup=100,
            )
            ret = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": "step",
                },
            }
            return ret
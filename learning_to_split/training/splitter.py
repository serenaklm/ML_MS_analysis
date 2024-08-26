import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from dataloader import get_dataloader
from utils import compute_marginal_z_loss, compute_y_given_z_loss

def _train_splitter_single_epoch(splitter, predictor, total_loader, test_loader, opt, config_dict):
    
    """
        train the splitter for one epoch    
    """

    stats = {}
    for k in ['loss_ratio', 'loss_gap', 'loss']: stats[k] = []
    
    for batch_total, batch_test in zip(total_loader, test_loader):

        # Regularity constraints
        mz_binned_total = batch_total[0]
        FP_total = batch_total[1]

        # Get the predicted logits
        logit_total = splitter(mz_binned_total)
        prob_total = F.softmax(logit_total, dim = -1)[:, 1]

        # This loss ensures that the training size and testing size are compariable.
        loss_ratio_total, ratio_total = compute_marginal_z_loss(prob_total, config_dict["train_ratio"])
        
        # This loss ensures that the marginal distributions of the label are
        # compariable in the training split and in the testing split
        loss_balance, ptrain_y, ptest_y = compute_y_given_z_loss(prob_total, FP_total)

        print("okay i am here")
        a = z 


    #     # Add standard loss to the stats
    #     stats["loss_ratio"].append(loss_ratio_total.item())

    #     # We now get the loss labels to train our splitter 
    #     smiles_test = batch_test[0]
    #     mz_binned_test = batch_test[1]
    #     logit_test = splitter(smiles_test, mz_binned_test)

    #     with torch.no_grad():

    #         smiles_test_emb, mz_binned_test_emb = predictor(smiles_test, mz_binned_test)
    #         test_emb = torch.concat([smiles_test_emb, mz_binned_test_emb], dim = 0)
    #         labels = torch.arange(smiles_test_emb.shape[0])
    #         labels = torch.concat([labels, labels], dim = 0).to(args.device)
    #         output = INFO_NCE_LOSS_NO_REDUCTION(test_emb, labels)

    #         output = output["loss"]["losses"]
    #         output = output.view(smiles_test_emb.shape[0], -1).contiguous() 
    #         output = torch.mean(output, dim = -1) 
        
    #     loss_gap = F.cross_entropy(logit_test, correct)
    #     stats["loss_gap"].append(loss_gap.item())

    #     # compute overall loss and update the parameters
    #     w_sum = args.w_gap + args.w_ratio # Assumes that each type is equally important

    #     loss = (loss_gap * args.w_gap
    #             + loss_ratio_total * args.w_ratio) / w_sum

    #     stats["loss"].append(loss.item())
    #     optim_step(splitter, opt, loss, args)

    # # Consolidate across the batches
    # for k, v in stats.items():
    #     stats[k] = torch.mean(torch.tensor(v)).item()

    # return stats


def train_splitter(splitter: nn.Module,
                   predictor: nn.Module,
                   data: Dataset,
                   test_indices: list[int],
                   opt: dict,
                   config_dict: dict,
                   verbose: bool = False):
    """
        Train the splitter to
        1. (on the test split) move wrong predictions to the train
        split and keep the incorrect predictions in the test split;
        2. (on all data) satisfy the regularity constraints
    """

    splitter.train()
    predictor.eval()

    # total_loader samples from the entire dataset
    # We use its samples to enforce the regularity constraints
    total_loader = get_dataloader(data, False, batch_size = config_dict["batch_size"],
                                  num_workers = config_dict["num_workers"], config_dict= config_dict,
                                  sampler = RandomSampler(data, replacement = True,
                                                          num_samples = config_dict["batch_size"] * config_dict["num_batches"]))

    test_data = Subset(data, indices = test_indices)
    test_loader = get_dataloader(test_data, False, batch_size = config_dict["batch_size"],
                                 num_workers = config_dict["num_workers"], config_dict= config_dict,
                                  sampler = RandomSampler(test_data, replacement = True,
                                                          num_samples = config_dict["batch_size"] * config_dict["num_batches"]))

    ep, loss_list = 0, []  # Keep track of the loss over the past 5 epochs

    while True:

        train_stats = _train_splitter_single_epoch(splitter, predictor, total_loader, test_loader, opt, config_dict)
        cur_loss = train_stats["loss"]

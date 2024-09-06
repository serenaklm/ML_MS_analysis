import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from dataloader import get_dataloader
from utils import to_tensor, compute_marginal_z_loss, compute_y_given_z_loss, optim_step, print, write_json

def print_splitter_stat(stats, i, output_path):

    print(f"| splitter ep {i} "
          f"loss {stats['loss']:>6.4f} "
          f"(gap {stats['loss_gap']:>6.4f} "
          f"ratio {stats['loss_ratio']:>6.4f} "
          f"label {stats['loss_balance']:>6.4f})",
          flush=True)

    write_json(stats, output_path)
    
def _train_splitter_single_epoch(splitter, predictor, total_loader, test_loader, opt, config_dict):
    
    """
        train the splitter for one epoch    
    """

    stats = {}
    for k in ['loss_ratio', 'loss_balance', 'loss_gap', 'loss']: stats[k] = []

    # Get the losses
    cs_loss = nn.CosineSimilarity()
    ce_loss = nn.CrossEntropyLoss() 

    for batch_total, batch_test in zip(total_loader, test_loader):

        # Regularity constraints
        mz_binned_total = batch_total[0]
        FP_total = to_tensor(batch_total[1]).to(config_dict["device"])

        # Get the predicted logits
        logit_total = splitter(mz_binned_total, FP_total)
        prob_total = F.softmax(logit_total, dim = -1)[:, 1]

        # This loss ensures that the training size and testing size are compariable.
        loss_ratio_total, _ = compute_marginal_z_loss(prob_total, config_dict["train_ratio"])
        
        # This loss ensures that the marginal distributions of the label are
        # compariable in the training split and in the testing split
        loss_balance_total, _, _ = compute_y_given_z_loss(prob_total, FP_total)

        # Add standard loss to the stats
        stats["loss_ratio"].append(loss_ratio_total.item())
        stats['loss_balance'].append(loss_balance_total.item())

        # We now get the loss labels to train our splitter 
        mz_binned_test = batch_test[0]
        FP_test = to_tensor(batch_test[1]).to(config_dict["device"])
        logit_test = splitter(mz_binned_test, FP_test)

        with torch.no_grad():

            FP_pred_test = predictor(mz_binned_test)
            cs = (cs_loss(FP_pred_test, FP_test) + 1.0) / 2
            
        # The higher the cs score, the more it is "correct"
        # Hence it should be assigned to the train set 
        correct = torch.stack([1.0 - cs, cs], dim = -1)
        
        loss_gap = ce_loss(logit_test, correct)
        stats["loss_gap"].append(loss_gap.item())

        # compute overall loss and update the parameters
        w_sum = config_dict["w_gap"] + config_dict["w_ratio"] + config_dict["w_balance"]

        loss = (loss_gap * config_dict["w_gap"]
                + loss_ratio_total * config_dict["w_ratio"]
                + loss_balance_total * config_dict["w_balance"]) / w_sum

        stats["loss"].append(loss.item())
        optim_step(splitter, opt, loss, config_dict)

    # Consolidate across the batches
    for k, v in stats.items():
        stats[k] = torch.mean(torch.tensor(v)).item()

    return stats

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

        if len(loss_list) == 5:
            if cur_loss > sum(loss_list) / 5.0 - config_dict['convergence_threshold']:
                # break if the avg loss in the past 5 iters doesn't improve beyond a defined threshold
                break

        if len(loss_list) == config_dict["patience"]:
            if cur_loss > sum(loss_list) / config_dict["patience"] - config_dict["convergence_threshold"]:

                # break if the avg loss in the past `patience` iters doesn't improve beyond a defined threshold
                break

            loss_list.pop(0)

        loss_list.append(cur_loss)

        progress_message =  f'train_splitter ep {ep}, loss {cur_loss:.4f}'
        print(progress_message, end="\r", flush=True, time=True)

        # Sanity check
        ep += 1
        if ep == 100:
            warnings.warn("Splitter (inner-loop) training fails to converge within 100 epochs.")

        elif ep == 1000:
            raise Error("Splitter (inner-loop) training fails to converge within 1000 epochs.")
        
    # Clear the progress
    # Note that we need to overwrite the time stamps too.
    print(" " * (20 + len(progress_message)), end="\r", time=False)

    # Print the last status
    print_splitter_stat(train_stats, ep, os.path.join(config_dict["results_folder"], "splitter_stats.json"))
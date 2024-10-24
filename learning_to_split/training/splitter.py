import warnings

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from utils import print, optim_step

def compute_marginal_z_loss(mask, tar_ratio, no_grad = False):

    '''
        Compute KL div between the splitter's z marginal and the prior z margional
        Goal: the predicted training size need to be tar_ratio * total_data_size
    '''

    cur_ratio = torch.mean(mask)
    cur_z = torch.stack([cur_ratio, 1 - cur_ratio])  # train split, test_split

    tar_ratio = torch.ones_like(cur_ratio) * tar_ratio
    tar_z = torch.stack([tar_ratio, 1 - tar_ratio])

    loss_ratio = F.kl_div(torch.log(cur_z), tar_z, reduction='batchmean')

    if not torch.isfinite(loss_ratio):
        loss_ratio = torch.ones_like(loss_ratio)

    if no_grad:
        loss_ratio = loss_ratio.item()

    return loss_ratio, cur_ratio.item()

def compute_y_given_z_loss(mask, FP, no_grad = False):
    
    '''
      conditional marginal p(y | z = 1) need to match p(y | z = 0)
    '''

    p_given_train, p_given_test, p_original = [],[], [] 

    for p in range(FP.shape[1]):

        y = FP[:, p]
        y_given_train, y_given_test, y_original = [], [], []

        y_i = (y == 0).float()

        y_given_train.append(torch.sum(y_i * mask) / torch.sum(mask))
        y_given_test.append(torch.sum(y_i * (1 - mask)) / torch.sum(1 - mask))
        y_original.append(torch.sum(y_i) / len(y))

        y_given_train = torch.stack(y_given_train)
        y_given_test = torch.stack(y_given_test)
        y_original = torch.stack(y_original).detach()

        p_given_train.append(y_given_train)
        p_given_test.append(y_given_test)
        p_original.append(y_original)
    
    # Get the loss now 
    p_given_train = torch.stack(p_given_train)
    p_given_test = torch.stack(p_given_test)
    p_original = torch.stack(p_original)

    loss_p_train_marginal = F.kl_div(torch.log(p_given_train), p_original, reduction = 'batchmean')
    loss_p_test_marginal = F.kl_div(torch.log(p_given_test), p_original, reduction = 'batchmean')
    loss_p_marginal = 1/2 * (loss_p_train_marginal + loss_p_test_marginal)

    if not torch.isfinite(loss_p_marginal):
        loss_p_marginal = torch.ones_like(loss_p_marginal)

    loss_p_marginal = torch.where(loss_p_marginal > 1e3, torch.tensor(1.0), loss_p_marginal)

    if no_grad:
        loss_p_marginal = loss_p_marginal.item()

    return loss_p_marginal

def _train_splitter_single_epoch(splitter, predictor, loader, test_loader, opt, config):
    

    stats = {}
    for k in ["loss_ratio", "loss_balance", "loss_gap", "loss"]: stats[k] = []
    
    for batch, batch_test in zip(loader, test_loader):

        logit = splitter(batch)
        FP = batch["FP"]
        prob = F.softmax(logit, dim = -1)[:, 1]

        # Ensures that the ratio are compatible
        loss_ratio, _ = compute_marginal_z_loss(prob, config["model"]["train_params"]["train_ratio"])

        # Ensures p(y | z = 1) need to match p(y | z = 0)
        loss_balance = compute_y_given_z_loss(prob, FP)

        # predictor's correctness as a loss 
        logit_test = splitter(batch_test)

        with torch.no_grad():

            FP_test = batch_test["FP"]
            FP_pred_test = predictor(batch_test)
            score = (F.cosine_similarity(FP_test, FP_pred_test) + 1.0) / 2.0 # Higher the score the more "correct" it is
            score_test = torch.concat([(1.0 - score)[:, None], score[:, None]], dim = -1)

        loss_gap = F.cross_entropy(logit_test, score_test) # Move the mistake to the test 

        # Get the combined loss 
        w_ratio = config["model"]["train_params"]["w_ratio"]
        w_balance = config["model"]["train_params"]["w_balance"]
        w_gap = config["model"]["train_params"]["w_gap"]
        w_sum = w_ratio + w_balance + w_gap 

        loss = (w_ratio * loss_ratio + w_balance * loss_balance + w_gap * loss_gap) / w_sum 

        # Optimize
        optim_step(splitter, opt, loss, config)

        # Update the stats
        stats["loss_ratio"].append(loss_ratio.item())
        stats["loss_balance"].append(loss_balance.item())
        stats["loss_gap"].append(loss_gap.item())
        stats["loss"].append(loss.item())

    for k, v in stats.items():
        stats[k] = torch.mean(torch.tensor(v)).item()

    return stats

def train_splitter(splitter: nn.Module,
                   predictor: nn.Module,
                   data: Dataset,
                   test_indices: list[int],
                   opt: dict,
                   config: dict,
                   verbose: bool = False):
    
    splitter.train()
    predictor.eval()

    # Get dataloaders
    batch_size = config["dataprocessing"]["batch_size"]
    num_batches = config["dataprocessing"]["num_batches"]
    num_workers = config["dataprocessing"]["num_workers"]

    test_data = Subset(data, indices = test_indices)
    loader = DataLoader(data, batch_size = batch_size,
                              sampler = RandomSampler(data, replacement = True,
                                                      num_samples = batch_size * num_batches),
                              num_workers = num_workers)
    
    test_loader = DataLoader(test_data, batch_size = batch_size,
                             sampler = RandomSampler(test_data, replacement = True,
                                                     num_samples = batch_size * num_batches),
                             num_workers = num_workers)
    
    ep, loss_list = 0, []

    while True:

        train_stats = _train_splitter_single_epoch(splitter, predictor, loader, test_loader, opt, config)
        cur_loss = train_stats["loss"]

        patience = config["model"]["train_params"]["patience"]
        convergence_threshold = config["model"]["train_params"]["convergence_threshold"]

        if len(loss_list) == patience:
            if cur_loss > sum(loss_list) / float(patience) - convergence_threshold:
                break

            loss_list.pop(0)

        loss_list.append(cur_loss)

        progress_message =  f'train_splitter ep {ep}, loss {cur_loss:.4f}'
        print(progress_message, end = "\r", flush = True, time = True)

        ep +=1  
        if ep == 100:
            warnings.warn("Splitter (inner-loop) training fails to converge within 100 epochs.")

        elif ep == 1000:
            raise Error("Splitter (inner-loop) training fails to converge within 1000 epochs.")

    print(" " * (20 + len(progress_message)), end="\r", time=False)
    print(" " * (20 + len(train_stats)), end="\r", time=False)

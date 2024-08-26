import torch
import torch.nn.functional as F

from utils import to_tensor

def compute_marginal_z_loss(mask, tar_ratio, no_grad=False):

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


def compute_y_given_z_loss(mask, y, no_grad=False):

    '''
      conditional marginal p(y | z=1) need to match p(y | z=0)
    '''

    num_bits = y.shape[-1] 
    y_given_train, y_given_test, y_original = [], [], []

    for i in range(num_bits):
        
        y_1 = (y[:, i] == 1).float()

        y_1_given_train = torch.sum(y_1 * mask) / torch.sum(mask) # p(y|z=1)
        y_1_given_test = torch.sum(y_1 * (1 - mask)) / torch.sum(1 - mask) # p(y|z=0)
        y_1_original = torch.sum(y_1) / len(y)  # p(y)

        y_given_train.append(torch.stack([y_1_given_train, 1.0 - y_1_given_train])) # assume binary
        y_given_test.append(torch.stack([y_1_given_test, 1.0 - y_1_given_test]))
        y_original.append(torch.stack([y_1_original, 1.0 - y_1_original]))

    y_given_train = torch.stack(y_given_train)
    y_given_test = torch.stack(y_given_test)
    y_original = torch.stack(y_original).detach()
    
    loss_y_marinal_train =  F.kl_div(torch.log(y_given_train), y_original,
                                     reduction='batchmean')

    loss_y_marinal_test =  F.kl_div(torch.log(y_given_test), y_original,
                                    reduction='batchmean')
    
    loss_y_marginal = loss_y_marinal_train + loss_y_marinal_test

    if not torch.isfinite(loss_y_marginal):
        loss_y_marginal = torch.ones_like(loss_y_marginal)

    if no_grad:
        loss_y_marginal = loss_y_marginal.item()

    return loss_y_marginal, y_given_train.tolist()[-1], y_given_test.tolist()[-1]
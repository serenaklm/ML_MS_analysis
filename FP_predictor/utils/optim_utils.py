import torch 
import torch.nn as nn 

def get_optim(model, config_dict):

    """
        return optimizer 
        To do: to add in a scheduler
    """
    
    result = {"optim": getattr(torch.optim, config_dict["optim"])(
              filter(lambda p: p.requires_grad, model.parameters()),
              lr = config_dict["lr"])}

    return result

def _valid_gradient(model: nn.Module):

    """
        Check that gradients are valid (not nan)
    """

    for p in model.parameters():

        if p.grad is not None:
            
            is_valid = torch.isfinite(p.grad).all()
            if not is_valid: # The gradients are not valid if any of them is nan or inf
                return False
            
    return True

def optim_step(model, opt_dict, loss, config_dict):

    """
        Apply the optimizer for one step. If the scheduler is present in the
        opt dict, update the scheduler after the optimizer.
    """

    opt_dict['optim'].zero_grad() # Zero grad before stepping
    loss.backward()

    if not _valid_gradient(model):
        
        opt_dict['optim'].zero_grad() # Zero out the gradient if the gradients contain nan or inf

    else:
        if config_dict["clip_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config_dict["clip_grad_norm"])

    opt_dict["optim"].step()

    if "lr_scheduler" in opt_dict:
        opt_dict["lr_scheduler"].step()
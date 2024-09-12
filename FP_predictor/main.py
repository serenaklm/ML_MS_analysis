import os
import wandb
import random
import argparse
import numpy as np
from datetime import datetime
from functools import partial
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *
from config import config_dict
from models import ModelFactory
from dataloader import CustomedDataset, get_DDP_dataloader

from dataloader.dataloader import get_DDP_dataloader

def get_predictions(rank, batch, model, loss_fn):

    # Get the partial function 
    to_tensor_p = partial(to_tensor, rank = rank)

    # Unpack the data
    mz_binned, FP = batch 
    mz_binned, FP = to_tensor_p(mz_binned), to_tensor_p(FP)

    # Get model's predictions 
    FP_pred = model(mz_binned, FP)

    # Get the loss 
    loss = loss_fn(FP_pred, FP)

    return FP_pred, FP, loss

def gather_outputs(outputs):

    group_size = torch.distributed.get_world_size(dist.group.WORLD)

    FP_pred = [torch.zeros_like(outputs[0]) for _ in range(group_size)]
    FP = [torch.zeros_like(outputs[1]) for _ in range(group_size)]
    loss = [torch.zeros_like(outputs[2]) for _ in range(group_size)]

    dist.all_gather(FP_pred, outputs[0])
    dist.all_gather(FP, outputs[1])
    dist.all_gather(loss, outputs[2])

    # Merge all together 
    FP_pred = torch.cat(FP_pred, dim = 0)
    FP = torch.cat(FP, dim = 0)
    loss = np.mean([l.item() for l in loss])

    return FP_pred, FP, loss

@torch.no_grad()
def test(rank, model, dataloader, get_predictions):
        
    # Iterate through the batches
    model.eval()
    n_batch, total_loss = 0, 0
    all_FP_pred, all_FP = [], []

    for batch in dataloader:
        
        FP_pred, FP, loss = gather_outputs(get_predictions(rank, batch))

        FP_pred = (torch.sigmoid(FP_pred) >= 0.5).int()
        all_FP_pred.extend(FP_pred.detach().cpu().numpy().tolist())
        all_FP.extend(FP.detach().cpu().numpy().tolist())

        n_batch += 1 
        total_loss += loss.item()

        model.train()

    avg_loss = round(total_loss / n_batch, 5)
    f_score = round(f1_score(all_FP, all_FP_pred, average = "samples"), 5)

    return avg_loss, f_score

def log_performance(rank, model, run, dataloader, get_predictions,
                    current_loss, best_loss, best_f_score,
                    global_step, config_dict):

    model_output_folder = os.path.join(config_dict["results_folder"], "models")
    logging_output_folder = os.path.join(config_dict["results_folder"], "messages")
    
    # Logging
    if rank == 0: 
        
        run.log({f"Training loss": current_loss})
        if not os.path.exists(model_output_folder): os.makedirs(model_output_folder)
        if not os.path.exists(logging_output_folder): os.makedirs(logging_output_folder)

    # Evaluate
    if global_step % config_dict["eval_iter"] == 0:
        
        test_loss, test_f_score = test(rank, model, dataloader, get_predictions)
        
        if rank == 0:

            run.log({f"Testing loss": test_loss, f"Testing F-score": test_f_score})
            save_model(model, os.path.join(model_output_folder, f"model_step_{global_step}.pkl"))
        
            if test_loss < best_loss:

                best_loss = test_loss
                run.log({"Best testing loss": best_loss})
                
                save_model_msg = f"Saved model (loss) at step: {global_step}, testing loss: {test_loss}, testing f-score: {test_f_score}"
                write_message(save_model_msg, os.path.join(logging_output_folder, "saved_model_loss.txt"))
                save_model(model, os.path.join(model_output_folder, "best_model_loss.pkl"))

            if test_f_score > best_f_score:

                best_f_score = test_f_score
                run.log({"Best testing f-score": best_f_score})
                
                save_model_msg = f"Saved model (F-score) at step: {global_step}, testing loss: {test_loss}, testing f-score: {test_f_score}"
                write_message(save_model_msg, os.path.join(logging_output_folder, "saved_model_f_score.txt"))
                save_model(model, os.path.join(model_output_folder, "best_model_f_score.pkl"))

    return best_loss, best_f_score

def main(rank, config_dict):
    
    # setup the process groups
    setup(rank, config_dict["world_size"])

    # Set up wandb on process 0 
    run = None
    if rank == 0: 
        run = wandb.init(project = "FP_predictor")
        run.name = "{}_{}".format(config_dict["model_name"], datetime.now().strftime("%d-%m-%Y-%H-%M"))
        run.config.update(config_dict)

    # Get the model
    model = ModelFactory.get_model(config_dict, predictor = True)
    model.to(rank)
    model = DDP(model, device_ids = [rank], output_device = rank)

    # Get the dataloaders
    data = CustomedDataset(config_dict["data_file_path"], config_dict)
    assert config_dict["train_ratio"] > 0 and config_dict["train_ratio"] < 1.0 
    train_size = int(len(data) * config_dict["train_ratio"])
    val_size   = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])
    
    train_dataloader = get_DDP_dataloader(rank, train_data, config_dict)
    val_dataloader = get_DDP_dataloader(rank, val_data, config_dict)

    # Get the optimizer 
    BCE_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr = config_dict["lr"], weight_decay = config_dict["l2_weight"])

    # Get some partial functions
    get_predictions_p = partial(get_predictions, model = model, loss_fn = BCE_loss)
    log_performance_p = partial(log_performance, model = model, run = run, dataloader = val_dataloader, 
                                get_predictions = get_predictions_p, 
                                config_dict = config_dict)

    # Run and do some logging  
    best_loss, best_f_score, global_step = 1e3, 0, 0

    for epoch in range(config_dict["n_epochs"]):

        train_dataloader.sampler.set_epoch(epoch)       

        # Iterate through the batches
        for batch in train_dataloader:

            model.train()

            _, _, loss = get_predictions_p(rank, batch)

            # Step and train
            loss.backward()
            optimizer.step()

            # Update the progress bar
            loss = loss.item()
            global_step += 1

            # Logging 
            best_loss, best_f_score = log_performance_p(rank = rank, current_loss = loss, 
                                                        best_loss = best_loss, 
                                                        best_f_score = best_f_score,
                                                        global_step = global_step)

    if rank == 0: run.finish()
    cleanup()

if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description = "Training a simple MLP model for FP prediction")

    # Model parameters 
    parser.add_argument("--model_name", type = str, default = "MLP", help = "Name of model")
    parser.add_argument("--dropout_rate", type = float, default = 0.2, help = "Dropout rate (default: 0.2)")
    parser.add_argument("--FP_type", type = str, default = "maccs", help = "The type of FP that we are predicting (default: maccs)")

    # Training parameters
    parser.add_argument("--n_epochs", type = int, default = 500, help = "Number of training epochs")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size")
    parser.add_argument("--num_workers", type = int, default = 2, help = "Number of workers to process the data")

    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning rate (default: 1e-4)")
    parser.add_argument("--l2_weight", type = float, default = 1e-4, help = "Weight for L2 normalization (default: 1e-4)")

    args = parser.parse_args()

    # Get the default parameters from the config file
    config_dict["input_dim"] = int(math.ceil(config_dict["max_mass"] / config_dict["granularity"]))

    config_dict["results_folder"] = os.path.join(config_dict["main_results_folder"], config_dict["FP_pred_results_folder"], "{}_{}".format(args.model_name, datetime.now().strftime("%d-%m-%Y-%H-%M")))
    if not os.path.exists(config_dict["results_folder"]): os.makedirs(config_dict["results_folder"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config_dict["device"] = device

    # Set a seed for reproducibility  
    seed = config_dict["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Log parameters and run main function
    write_dict(config_dict, os.path.join(config_dict["results_folder"], "args.json"), skip = ["device"])

    world_size = torch.cuda.device_count()
    args.world_size = world_size
    config_dict.update(args.__dict__)

    mp.spawn(main, args = (config_dict,),
                   nprocs = world_size,
                   join = True)
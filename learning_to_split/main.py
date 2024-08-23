import os 
import copy
import argparse 
from datetime import datetime
from torch.utils.data import Dataset, Subset

from utils import *
from config import *
from dataloader import CustomedDataset, get_DDP_dataloader

def learning_to_split(args: argparse.Namespace,
                      data: Dataset,
                      verbose: bool = True):

    """
        ls: learning to split
        trains a splitter to split the dataset into training / testing such that 
        a learnt predictor will perform poorly on the testing data

        return_order = ['train_data', 'test_data', 'train_indices', 'test_indices', 'splitter', 'predictor'])
    """

    num_no_improvements = 0
    best_gap, best_split = -1, None  # The bigger the gap, the better (more challenging) the split.

    # Some sanity check 
    assert args.train_ratio > 0.0 and args.train_ratio < 1.0, "Training ratio needs to be between 0.0 and 1.0."

    # Initialize the spliiter and the optimizer 
    splitter = ModelFactory.get_model(args, splitter = True)
    opt = get_optim(splitter, args)


    

if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description = "Getting testing results of trained model")

    # Data settings 
    parser.add_argument("--dataset", type = str, default = "canopus", help = "The dataset")

    # Model parameters 
    parser.add_argument("--model_name", type = str, default = "MLP", help = "Name of model")
    parser.add_argument("--input_dim", type = int, default = 2000, help = "Embedding dimension of the input MS (default: 2000)")
    parser.add_argument("--emb_dim", type = int, default = 1024, help = "Embedding dimension of the tokens (default: 1024)")
    parser.add_argument("--hidden_dim", type = int, default = 4096, help = "Hidden embedding dimension of the tokens (default: 4096)")
    parser.add_argument("--dropout_rate", type = float, default = 0.2, help = "Dropout rate (default: 0.2)")

    # Training parameters 
    parser.add_argument("--n_outer_loops", type = int, default = 500, help = "Maximum number of outer loops epoch")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size")
    parser.add_argument("--num_batches", type = int, default = 100, help = "Number of batches for each epoch")
    parser.add_argument("--num_workers", type = int, default = 2, help = "Number of workers to process the data")
    parser.add_argument("--train_ratio", type = float, default = 0.8, help = "Percentage of data allocated for training")

    parser.add_argument("--optim", type = str, default = "Adam", help = "The optimizer to use. Refer to torch.optim for list of options")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning rate")
    parser.add_argument("--clip_grad_norm", type = float, default = 1, help = "To clip the gradients, set as 0 if do not clip")
    parser.add_argument("--patience", type = int, default = 5, help = "Number of epochs to train the predictor when there's no improvement in val accuracy")
    parser.add_argument("--convergence_threshold", type = float, default = 1e-3, help = "Convergence threshold for the splitter training")

    parser.add_argument("--w_gap", type = int, default = 1, help = "Weight for the generalization gap loss")
    parser.add_argument("--w_ratio", type = int, default = 1, help = "Weight for the train/test ratio loss")

    args = parser.parse_args()

    # Get the default parameters from the config file
    args.data_file_path = data_file_path
    args.results_folder = os.path.join(results_folder, "{}_{}".format(args.model_name, datetime.now().strftime("%d-%m-%Y-%H-%M")))
    if not os.path.exists(args.results_folder): os.makedirs(args.results_folder)
    # args.collate_fn = collate_fn
    args.device = device

    # Log parameters and run learning to split
    write_args(args, os.path.join(args.results_folder, "args.json"), skip = ["collate_fn", "device"])
    data = CustomedDataset(args.data_file_path)
    learning_to_split(args, data)

import os 
import copy
import argparse 
from datetime import datetime
from torch.utils.data import Dataset, Subset

from utils import *
from config import config_dict

from models.build import ModelFactory
from dataloader import CustomedDataset
from training import split_data, train_predictor, test_predictor, train_splitter

def save_data(data, best_split):

    # Train indices, test indices
    train_indices, test_indices = best_split['train_indices'], best_split['test_indices']

    # Train and test data
    train_data = Subset(data, indices = best_split['train_indices'])
    test_data = Subset(data, indices = best_split['test_indices'])

    # Get the weights of the model
    splitter_weights = best_split["splitter"]
    predictor_weights = best_split["predictor"]

    # Get splits stats 
    stats = {str(k):v for k, v in best_split.items() if k not in ["splitter", "predictor"]}

    pickle_data(train_data, os.path.join(args.results_folder, "train.pkl"))
    pickle_data(test_data, os.path.join(args.results_folder, "test.pkl"))
    pickle_data(train_indices, os.path.join(args.results_folder, "train_indices.pkl"))
    pickle_data(test_indices, os.path.join(args.results_folder, "test_indices.pkl"))
    save_model(splitter_weights, os.path.join(args.results_folder, "splitter.pth"))
    save_model(predictor_weights, os.path.join(args.results_folder, "predictor.pth"))
    
    write_json(stats, os.path.join(args.results_folder, "splits_stats.json"))

def learning_to_split(config_dict: dict,
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
    assert config_dict["train_ratio"] > 0.0 and config_dict["train_ratio"] < 1.0, "Training ratio needs to be between 0.0 and 1.0."

    # Initialize the spliiter and the optimizer 
    splitter = ModelFactory.get_model(config_dict, splitter = True)
    opt = get_optim(splitter, config_dict)

    # Start training
    for outer_loop in range(config_dict["n_outer_loops"]):

        # Get the predictor
        predictor = ModelFactory.get_model(config_dict, predictor = True)

        # Step 1: Split the dataset using the Splitter
        # We start with random split for the first iteration
        random_split = True if outer_loop == 0 else False
        split_stats, train_indices, test_indices = split_data(data, splitter, config_dict, random_split) 
        
        # Step 2: train and test the predictor
        val_score = train_predictor(data = data, train_indices = train_indices,
                                    predictor = predictor, config_dict = config_dict)
        
        test_score = test_predictor(data = data, test_indices = test_indices,
                                    predictor = predictor, config_dict= config_dict)

        if verbose: print_split_status(outer_loop, split_stats, val_score, test_score)

        # Save the splitter and predictor if it produces a more challenging split
        gap = val_score - test_score

        if gap > best_gap:
            
            best_gap, num_no_improvements = gap, 0

            best_split = {"splitter":      copy.deepcopy(splitter.state_dict()),
                          "predictor":     copy.deepcopy(predictor.state_dict()),
                          "train_indices": train_indices,
                          "test_indices":  test_indices,
                          "val_score":     val_score,
                          "test_score":    test_score,
                          "split_stats":   split_stats,
                          "outer_loop":    outer_loop,
                          "best_gap":      best_gap}
            
            # Write the states to a file 
            write_dict(best_split, os.path.join(config_dict["results_folder"], "best_split.json"),
                       skip = ["splitter", "predictor", "train_indices", "test_indices"])
            
        else: num_no_improvements += 1
        if num_no_improvements == args.patience: break

        # Step 3: Train the splitter based on performance from predictor
        train_splitter(splitter, predictor, data, test_indices, opt, config_dict,
                       verbose = verbose)

    # Done! 
    # Print the best split.
    if verbose:
        print("Finished!\nBest split:")
        print_split_status(best_split['outer_loop'], best_split['split_stats'],
                           best_split['val_score'], best_split['test_score'])

    # Save the relevant records
    save_data(data, best_split)

if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description = "Getting testing results of trained model")

    # Model parameters 
    parser.add_argument("--model_name", type = str, default = "MLP", help = "Name of model")
    parser.add_argument("--dropout_rate", type = float, default = 0.2, help = "Dropout rate (default: 0.2)")
    parser.add_argument("--FP_type", type = str, default = "maccs", help = "The type of FP that we are predicting (default: maccs)")

    # Training parameters
    parser.add_argument("--n_outer_loops", type = int, default = 500, help = "Maximum number of outer loops epoch")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size")
    parser.add_argument("--num_batches", type = int, default = 300, help = "Number of batches for each epoch")
    parser.add_argument("--num_workers", type = int, default = 2, help = "Number of workers to process the data")
    parser.add_argument("--train_ratio", type = float, default = 0.8, help = "Percentage of data allocated for training")

    parser.add_argument("--optim", type = str, default = "Adam", help = "The optimizer to use. Refer to torch.optim for list of options")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning rate")
    parser.add_argument("--clip_grad_norm", type = float, default = 1, help = "To clip the gradients, set as 0 if do not clip")
    parser.add_argument("--patience", type = int, default = 5, help = "Number of epochs to train the predictor when there's no improvement in val accuracy")
    parser.add_argument("--convergence_threshold", type = float, default = 1e-3, help = "Convergence threshold for the splitter training")

    parser.add_argument("--w_gap", type = int, default = 1, help = "Weight for the generalization gap loss")
    parser.add_argument("--w_ratio", type = int, default = 1, help = "Weight for the train/test ratio loss")
    parser.add_argument("--w_balance", type = int, default = 1, help = "Weight for the train/test label ratio loss")

    args = parser.parse_args()

    config_dict.update(args.__dict__)

    # Get the default parameters from the config file
    config_dict["input_dim"] = config_dict["max_length"]
    config_dict["results_folder"] = os.path.join(config_dict["results_folder"], "{}_{}".format(args.model_name, datetime.now().strftime("%d-%m-%Y-%H-%M")))
    if not os.path.exists(config_dict["results_folder"]): os.makedirs(config_dict["results_folder"])

    GPU_device_idx = config_dict["GPU_device_idx"]
    device = torch.device(f"cuda:{GPU_device_idx}") if torch.cuda.is_available() else torch.device("cpu")

    config_dict["device"] = device
    config_dict["output_dim"] = config_dict["FP_dim_mapping"][args.FP_type]

    # Log parameters and run learning to split
    write_dict(config_dict, os.path.join(config_dict["results_folder"], "args.json"), skip = ["device"])
    data = CustomedDataset(config_dict["data_file_path"])

    learning_to_split(config_dict, data)

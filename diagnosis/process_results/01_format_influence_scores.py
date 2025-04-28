import os
import pickle
from tqdm import tqdm 
from pathlib import Path

def load_pickle(path):

    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_data(data, path):
    
    with open(path, "wb") as f:
        pickle.dump(data, f)

def format_scores(folder):

    train_ids = load_pickle(os.path.join(folder, "train_ids.pkl"))
    test_ids = load_pickle(os.path.join(folder, "test_ids.pkl"))
    pairwise_scores = load_pickle(os.path.join(folder, "EK-FAC_scores.pkl"))["all_modules"]

    formatted_scores = {}

    for test_idx in tqdm(range(pairwise_scores.shape[0])):
        
        current_rec = {} 

        for train_idx in tqdm(range(pairwise_scores.shape[1])):
            current_rec[train_ids[train_idx].split("/")[-1].replace(".ms", "").replace(".pkl", "")] = pairwise_scores[test_idx, train_idx].item()    
        
        formatted_scores[test_ids[test_idx].split("/")[-1].replace(".ms", "").replace(".pkl", "")] = current_rec
    
    return formatted_scores

if __name__ == "__main__":

    TOP_K = 1000
    CACHE_FOLDER = "./cache"
    if not os.path.exists(CACHE_FOLDER): os.makedirs(CACHE_FOLDER)

    baseline_models_folder = "../../FP_prediction/baseline_models/best_models"
    mist_folder = "../../FP_prediction/mist/best_models"

    RESULTS_FOLDER = []

    for folder in [mist_folder]:
        for dataset in os.listdir(folder):
            for checkpoint in os.listdir(os.path.join(folder, dataset)):
                RESULTS_FOLDER.append(os.path.join(folder, dataset, checkpoint))

    # Iterate through all the folders and get the formatted scores 
    for checkpoint in RESULTS_FOLDER:
        
        # Check that the scores exist 
        score_path = Path(checkpoint) / "EK-FAC_scores.pkl"
        output_path = Path(checkpoint) / "EK-FAC_scores_formatted.pkl"
        if not os.path.exists(score_path): continue 
        if os.path.exists(output_path): continue

        # Get the formatted score 
        print(f"Formatting the scores for {checkpoint} now")
        formatted_scores = format_scores(checkpoint)
        pickle_data(formatted_scores, output_path)
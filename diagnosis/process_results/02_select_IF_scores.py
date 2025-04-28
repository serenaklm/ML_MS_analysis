import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def load_pickle(path):

    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_data(data, path):
    
    with open(path, "wb") as f:
        pickle.dump(data, f)

def select_datapoints(distances, k):
    
    selected_indices = []

    # Step 1: Randomly pick the first point
    first_idx = np.random.choice(distances.shape[0])
    selected_indices.append(first_idx)

    # Step 2: Select the next k-1 points
    for _ in tqdm(range(1, k)):
        
        # For each point, find its distance to the nearest selected point
        min_distances = np.min(distances[:, selected_indices], axis=1)
        
        # Set distance of already selected points to -inf to avoid reselection
        min_distances[selected_indices] = -np.inf
        
        # Select the point with the maximum of these minimum distances
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx.item())

    return selected_indices


if __name__ == "__main__":

    results_folder = "../../FP_prediction/mist/best_models/"
    all_checkpoints = [] 

    for dataset in os.listdir(results_folder):
        if "sieved" in dataset: continue 
        for checkpoint in os.listdir(os.path.join(results_folder, dataset)):
            all_checkpoints.append(os.path.join(results_folder, dataset, checkpoint))

    all_checkpoints = [f for f in all_checkpoints if os.path.exists(os.path.join(f, "EK-FAC_scores.pkl"))]

    for CHECKPOINT in all_checkpoints:

        print(f"Processing {CHECKPOINT} now ")
        train_ids = load_pickle(os.path.join(CHECKPOINT, "train_ids.pkl"))
        scores = load_pickle(os.path.join(CHECKPOINT, "EK-FAC_scores.pkl"))["all_modules"].T.numpy()

        # Precompute the distance matrix 
        distances = pairwise_distances(scores, scores)
        
        for ratio in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:

            # Get the output path 
            ratio_int = int(ratio * 100)
            output_path = os.path.join(CHECKPOINT, f"selected_train_{ratio_int}.pkl")

            if os.path.exists(output_path): continue 

            n_datapoints = int(ratio * len(train_ids))
            selected_train_idx = select_datapoints(distances, k = n_datapoints)
            selected_train_ids = [train_ids[i] for i in selected_train_idx]

            # Save the records now 
            pickle_data(selected_train_ids, output_path)
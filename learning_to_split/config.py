import torch 

# Data folder 
data_file_path = "../data/final_w_classyfire_annotations"
results_folder = "../LS_results"

# GPU device 
GPU_device_idx = 0
device = torch.device(f"cuda:{GPU_device_idx}") if torch.cuda.is_available() else torch.device("cpu")


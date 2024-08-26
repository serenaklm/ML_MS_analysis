

config_dict = {# Data folder 
               "data_file_path" : "../data/final/final_data.msp",
               "results_folder" : "../LS_results",                  
               
               # Get the mapping of output dimension to the FP that we are predicting
               "FP_dim_mapping" : {"maccs": 167,
                                   "morgan4_256": 256,
                                   "morgan4_1024": 1024, 
                                   "morgan4_2048": 2048,
                                   "morgan4_4096": 4096,
                                   "morgan6_256": 256,
                                   "morgan6_1024": 1024, 
                                   "morgan6_2048": 2048,
                                   "morgan6_4096": 4096},

                # Get dimensions of the MLP model
                "MLP_model_dim": {"emb_dim": 1024, 
                                  "hidden_dim": 4096},

                # Max length of binned MS
                "max_length" : 1000,
                
                # GPU device 
                "GPU_device_idx": 0,
                
                # Class threshold that puts the label into class 1
                "class_threshold": 0.5}
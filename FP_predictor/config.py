config_dict = {# Data folders
               "data_file_path" : "../data/final/final_data.msp",
               "main_results_folder": "../results",
               "FP_pred_results_folder" : "../FP_prediction_results",                  
               
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

                # Set parameters for binning of MS
                "max_mass" : 1000,
                "granularity": 0.1,
                
                # Random seed 
                "seed": 17,

                # Get random subset of data for training 
                "train_ratio": 0.8,
                
                # For evaluation
                "eval_iter": 100, 

                # Class threshold that puts the label into class 1
                "class_threshold": 0.5}

import os 
import json
import shutil
import subprocess
from matchms.importing import load_from_mgf

if __name__ == "__main__":


    project_folder =  "C:\\Users\\klingmin\\Desktop\\projects\\analyze_sirius_fragmentation_tree\\project"
    temp_folder = "C:\\Users\\klingmin\\Desktop\\projects\\analyze_sirius_fragmentation_tree\\temp"
    data_folder =  'C:\\Users\\klingmin\\Desktop\\projects\\analyze_sirius_fragmentation_tree\\individual_MS_alpha_0.0\\'

    for alpha in [0.0, 0.05, 0.10, 0.15]:

        folder = f"C:\\Users\\klingmin\\Desktop\\projects\\analyze_sirius_fragmentation_tree\\individual_MS_alpha_{alpha}\\"
        results_folder = f"./results_alpha_{alpha}"
        if not os.path.exists(results_folder): os.makedirs(results_folder)
        i = 0 

        for f in os.listdir(folder):

            input_path = data_folder + f 
            s = [f for f in load_from_mgf(input_path)]
            adduct = s[0].metadata["adduct"]

            subprocess.run(["sirius", "--input", input_path, "--output", project_folder, "formula"], shell=True)
            subprocess.run(["sirius", "--input", project_folder, "ftree-export", "--json", "--output", temp_folder], shell=True)

            output_path = f.replace(".mgf", ".json")
            output_path = os.path.join(results_folder, output_path)

            # Move the folder 
            file_in_temp =  [n for n in os.listdir(temp_folder)]
            assert len(file_in_temp) == 1 
            file_in_temp = os.path.join(temp_folder, file_in_temp[0])

            shutil.move(file_in_temp, output_path) 

            # Delete the folders 
            shutil.rmtree(project_folder) 
            shutil.rmtree(temp_folder) 

            # Update 
            i += 1 
            if i == 10: break 
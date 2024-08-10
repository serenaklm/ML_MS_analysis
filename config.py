import os

# URL for dataset donwload 
MONA_url = "https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/7609a87b-5df1-4343-afe9-2016a3e79516"
massbank_url = "https://github.com/MassBank/MassBank-data/releases/download/2024.06/MassBank.json"
GNPS_url = "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json"

# Folders
main_data_folder = "./data"
raw_data_folder = os.path.join(main_data_folder, "raw")
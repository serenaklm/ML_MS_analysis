import os
import wget

from config import *
from utils import download_data

if __name__ == "__main__":

    if not os.path.exists(raw_data_folder): os.makedirs(raw_data_folder)

    # Download MONA
    download_data(MONA_url, os.path.join(raw_data_folder, "MONA.json"))

    # Download MassBank
    download_data(massbank_url, os.path.join(raw_data_folder, "massbank.json"))

    # Download GNPS
    download_data(GNPS_url, os.path.join(raw_data_folder, "GNPS.json"))
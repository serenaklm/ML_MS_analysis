import os
import shutil

from config import *
from utils import download_data, unzip

if __name__ == "__main__":

    if not os.path.exists(raw_data_folder): os.makedirs(raw_data_folder)

    # Download the extra MS
    for link in extra_MS_url_list:
        filename = link.split("/")[-1].split("?")[0]
        download_data(link, os.path.join(raw_data_folder, filename))

    # Download GNPS
    for filename in GNPS_url_list:
        download_data(os.path.join(GNPS_main_url, filename), os.path.join(raw_data_folder, filename))

    # Download MassBank
    download_data(massbank_url, os.path.join(raw_data_folder, "massbank_NIST.msp"))

    # Download MONA
    if os.path.exists(os.path.join(raw_data_folder, "mona.msp")):
        output_path = os.path.join(raw_data_folder, "mona.msp")
        print(f"{output_path} already exists")
    else:
        download_data(mona_url, os.path.join(raw_data_folder, "mona.zip"))
        unzip(os.path.join(raw_data_folder, "mona.zip"), os.path.join(raw_data_folder, "mona"))
        shutil.move(os.path.join(raw_data_folder, "mona", "MoNA-export-LC-MS-MS_Positive_Mode.msp"),
                    os.path.join(raw_data_folder, "mona.msp"))
        os.rmdir(os.path.join(raw_data_folder, "mona"))
        os.remove(os.path.join(raw_data_folder, "mona.zip"))
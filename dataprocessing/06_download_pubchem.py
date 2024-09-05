import os
import requests
from tqdm import tqdm
import pubchempy as pcp
from bs4 import BeautifulSoup
import urllib.request, urllib.error, urllib.parse

from config import *
from utils import download_data, write_json, load_json


def get_webpage_list(url):

    response = urllib.request.urlopen(url)
    response = response.read().decode('UTF-8')
    soup = BeautifulSoup(response, "html.parser")
    file_list = [r["href"] for r in soup.find_all('a')]
    file_list = [f for f in file_list if f.endswith(".gz")]
    return file_list

def download_all_files(data_folder, file_list):

    for f in tqdm(file_list):

        path = os.path.join(data_folder, f)
        if os.path.exists(path): continue 
        download_data(pubchem_url + f, path)

if __name__ == "__main__": 

    # Create folder to download the data
    pubchem_folder = os.path.join(main_data_folder, "pubchem")
    if not os.path.exists(pubchem_folder): os.makedirs(pubchem_folder)
    
    # Download the list of files on pubchem
    file_list_path = os.path.join(pubchem_folder, "file_list.json")
    if not os.path.exists(file_list_path):
        file_list = get_webpage_list(pubchem_url)
        write_json(file_list, file_list_path)
    else: 
        file_list = load_json(file_list_path)

    # Download the data
    download_all_files(pubchem_folder, file_list)    


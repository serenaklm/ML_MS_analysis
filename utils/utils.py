import os 
import wget

def download_data(url, output_path):

    if os.path.exists(output_path): 
        print(f"{output_path} already exists")
    else:
        wget.download(url, out = output_path)
        print(f"downloaded {output_path}")
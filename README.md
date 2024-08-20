## This repo stores all the code that has been curated to understand the limits of ML for MS 

### Downloading datasets
``01_download.py`` downloads the necessary datasets for analysis. 
As of now, we are downloading the following datasets: 

1. MONA 
2. MassBank
3. GNPS

These records were downloaded on 08/10/2024.

### Format the datasets
``02_format_data.py`` formats all the datasets that we have downloaded in the previous step. This step includes standardizing some of the fields.

### Clean the datasets
``03_clean_data.py`` formats all the datasets that we have downloaded in the previous step. This step includes standardizing some of the fields.

### Merge the datasets 
``04_merge_data.py`` merge the records in all the dataset and assign the dataset name and new id_ to each record. 

### Get class annotation for each molecule 

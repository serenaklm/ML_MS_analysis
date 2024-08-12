import os

# URL for dataset donwload 
mona_url = "https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/873fbe29-4808-46d1-a4a3-a4134ac8c755"
massbank_url = "https://github.com/MassBank/MassBank-data/releases/download/2024.06/MassBank_NIST.msp"

GNPS_main_url = "https://gnps-external.ucsd.edu/gnpslibrary/"
GNPS_url_list = ['BERKELEY-LAB.mgf', 'BILELIB19.mgf', 'BIRMINGHAM-UHPLC-MS-POS.mgf',
                 'BMDMS-NP.mgf', 'CASMI.mgf', 'DRUGS-OF-ABUSE-LIBRARY.mgf', 'ECG-ACYL-AMIDES-C4-C24-LIBRARY.mgf',
                 'ECG-ACYL-ESTERS-C4-C24-LIBRARY.mgf', 'GNPS-COLLECTIONS-MISC.mgf',
                 'GNPS-COLLECTIONS-PESTICIDES-POSITIVE.mgf', 'GNPS-D2-AMINO-LIPID-LIBRARY.mgf',
                 'GNPS-EMBL-MCF.mgf', 'GNPS-FAULKNERLEGACY.mgf', 'GNPS-IOBA-NHC.mgf', 'GNPS-LIBRARY.mgf', 'GNPS-MSMLS.mgf',
                 'GNPS-NIH-CLINICALCOLLECTION1.mgf', 'GNPS-NIH-CLINICALCOLLECTION2.mgf',
                 'GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf',
                 'GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE.mgf', 'GNPS-NIH-SMALLMOLECULEPHARMACOLOGICALLYACTIVE.mgf',
                 'GNPS-NIST14-MATCHES.mgf', 'GNPS-NUTRI-METAB-FEM-POS.mgf', 'GNPS-PRESTWICKPHYTOCHEM.mgf',
                 'GNPS-SAM-SIK-KANG-LEGACY-LIBRARY.mgf', 'GNPS-SCIEX-LIBRARY.mgf', 'GNPS-SELLECKCHEM-FDA-PART1.mgf',
                 'GNPS-SELLECKCHEM-FDA-PART2.mgf', 'HCE-CELL-LYSATE-LIPIDS.mgf', 'HMDB.mgf', 'IQAMDB.mgf', 'LDB_POSITIVE.mgf', 'MIADB.mgf',
                 'MMV_POSITIVE.mgf', 'PNNL-LIPIDS-POSITIVE.mgf', 'PSU-MSMLS.mgf', 'RESPECT.mgf', 'SUMNER.mgf', 'UM-NPDC.mgf']

extra_MS_url_list = ["https://zenodo.org/records/11163381/files/20231031_nihnp_library_pos_all_lib_MSn.mgf?download=1",
                     "https://zenodo.org/records/11163381/files/20231130_mcescaf_library_pos_all_lib_MSn.mgf?download=1",
                     "https://zenodo.org/records/11163381/files/20231130_otavapep_library_pos_all_lib_MSn.mgf?download=1",
                     "https://zenodo.org/records/11163381/files/20240411_mcebio_library_pos_all_lib_MSn.mgf?download=1"]

# Folders
main_data_folder = "./data"
raw_data_folder = os.path.join(main_data_folder, "raw")
processed_data_folder = os.path.join(main_data_folder, "processed")
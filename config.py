import os

# URL for dataset donwload 
mona_url = "https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/873fbe29-4808-46d1-a4a3-a4134ac8c755"
massbank_url = "https://github.com/MassBank/MassBank-data/releases/download/2024.06/MassBank_NIST.msp"

GNPS_main_url = "https://external.gnps2.org/gnpslibrary/"
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

# GNPS_url_list = ["ALL_GNPS_NO_PROPOGATED.mgf"]

extra_MS_url_list = ["https://zenodo.org/records/11163381/files/20231031_nihnp_library_pos_all_lib_MSn.mgf?download=1",
                     "https://zenodo.org/records/11163381/files/20231130_mcescaf_library_pos_all_lib_MSn.mgf?download=1",
                     "https://zenodo.org/records/11163381/files/20231130_otavapep_library_pos_all_lib_MSn.mgf?download=1",
                     "https://zenodo.org/records/11163381/files/20240411_mcebio_library_pos_all_lib_MSn.mgf?download=1"]

# Folders
main_data_folder = "./data"
raw_data_folder = os.path.join(main_data_folder, "raw")
processed_data_folder = os.path.join(main_data_folder, "processed")
cleaned_data_folder = os.path.join(main_data_folder, "cleaned")
final_data_folder = os.path.join(main_data_folder, "final")

# Get list of adducts to include 
adducts = ["[M+H]+", "[M+Na]+", "[M+NH4]+", "M+H", "M+Na", "M+NH4", "[M+H]", "[M]+",
           "[M+H-H2O]+", "M-H2O+H", "M+H-H2O", "[M-H2O+H]+", "[M+H-2H2O]+"]

adducts_mapping = {"[M]+" : "[M]+",
                   "[M+H]+" : "[M+H]+", 
                   "M+H": "[M+H]+",
                   "[M+H]": "[M+H]+",
                   "[M+Na]+": "[M+Na]+",
                   "M+Na": "[M+Na]+",
                   "[M+NH4]+": "[M+NH4]+", 
                   "M+NH4": "[M+NH4]+", 
                   "[M+H-H2O]+": "[M+H-H2O]+", 
                   "M-H2O+H": "[M+H-H2O]+", 
                   "M+H-H2O": "[M+H-H2O]+", 
                   "[M-H2O+H]+": "[M+H-H2O]+", 
                   "[M+H-2H2O]+": "[M+H-2H2O]+"}

instruments_mapping = {'-Maxis HD qTOF': 'ESI-QTOF', 'ESI-QTOF': 'ESI-QTOF', '-Q-Exactive Plus Orbitrap Res 14k': 'ESI-QFT', '-Q-Exactive Plus Orbitrap Res 70k': 'ESI-QFT',
                       'APCI-Ion Trap': 'APCI-IT', 'APCI-Orbitrap': 'APCI-QFT', 'APCI-QQQ': 'APCI-QQ', 'APCI-qTof': 'APCI-QTOF', 'CI (MeOH)-IT/ion trap': 'CI-IT',
                       'CI-IT/ion trap': 'CI-IT', 'DI-ESI-Hybrid FT': 'ESI-QFT', 'DI-ESI-Ion Trap': 'ESI-IT', 'DI-ESI-Orbitrap': 'ESI-QFT',
                       'DI-ESI-Q-Exactive Plus': 'ESI-QFT', 'DI-ESI-QQQ': 'ESI-QQ', 'DI-ESI-qTof': 'ESI-QTOF', 
                       'DIRECT INFUSION NANOESI-ION TRAP-DIRECT INFUSION NANOESI-ION TRAP': 'ESI-IT', 'ESI or APCI-IT/ion trap': 'ESI-IT',
                       'APCI-ITFT': 'APCI-ITFT', 'LC-APCI-ITFT': 'APCI-ITFT', 'ESI-APCI-ITFT': 'APCI-ITFT', 'ESI-ESI-FTICR': 'ESI-FT', 'ESI-ESI-ITFT': 'ESI-ITFT', 'ESI-FAB-EBEB': 'FAB-EBEB',
                       'ESI-Flow-injection QqQ/MS': 'ESI-QQ', 'ESI-HCD': 'ESI-QFT', 'ESI-HPLC-ESI-TOF': 'LC-ESI-TOF', 'ESI-Hybrid FT': 'ESI-QFT',
                       'ESI-IT-FT/ion trap with FTMS': 'ESI-ITFT', 'ESI-IT/ion trap': 'ESI-IT', 'ESI-Ion Trap': 'ESI-IT', 'ESI-LC-APPI-QQ': 'LC-APPI-QQ',
                       'LC-ESI-IT': 'LC-ESI-IT', 'ESI-LC-ESI-IT': 'LC-ESI-IT', 'ESI-LC-ESI-ITFT': 'LC-ESI-ITFT', 'LC-ESI-ITFT': 'LC-ESI-ITFT', 'ESI-LC-ESI-ITTOF': 'LC-ESI-ITTOF', 'ESI-LC-ESI-Q': 'LC-ESI-Q',
                       'LC-ESI-QFT':'LC-ESI-QFT', 'ESI-LC-ESI-QFT': 'LC-ESI-QFT', 'LC-ESI-QQ':'LC-ESI-QQ', 'ESI-LC-ESI-QQ': 'LC-ESI-QQ', 'ESI-LC-ESI-QTOF': 'LC-ESI-QTOF', 'LC-ESI-QTOF': 'LC-ESI-QTOF', 'ESI-LC-Q-TOF/MS': 'LC-ESI-QTOF',
                       'ESI-Orbitrap': 'ESI-ITFT', 'ESI-Q-TOF': 'ESI-QTOF', 'ESI-QIT': 'ESI-QIT', 'ESI-QQQ': 'ESI-QQ', 'ESI-QqQ': 'ESI-QQ', 'ESI-UPLC-ESI-QTOF': 'LC-ESI-QTOF',
                       'ESI-qTOF': 'ESI-QTOF', 'ESI-qToF': 'ESI-QTOF', 'ESI-qTof': 'ESI-QTOF', 'FAB-BEqQ/magnetic and electric sectors with quadrupole': 'FAB-BEQQ',
                       'In-source CID-API': 'ESI-QQ', 'LC-APCI-qTof': 'LC-APCI-QTOF', 'LC-ESI- impact HD': 'LC-ESI-QTOF', 'LC-ESI-CID; Lumos': 'LC-ESI-ITFT',
                       'LC-ESI-CID; Velos': 'LC-ESI-ITFT', 'LC-ESI-HCD; Lumos': 'LC-ESI-ITFT', 'LC-ESI-HCD; Velos': 'LC-ESI-ITFT', 'LC-ESI-Hybrid FT': 'LC-ESI-QFT',
                       'LC-ESI-Hybrid Ft': 'LC-ESI-QFT', 'LC-ESI-ITFT-LC-ESI-ITFT': 'LC-ESI-ITFT', 'LC-ESI-ITTOF-LC-ESI-ITTOF': 'LC-ESI-ITTOF', 'LC-ESI-Ion Trap': 'LC-ESI-IT',
                       'LC-ESI-LCQ': 'LC-ESI-IT', 'LC-ESI-Maxis HD qTOF': 'LC-ESI-QTOF', 'LC-ESI-Maxis II HD Q-TOF Bruker': 'LC-ESI-QTOF', 'LC-ESI-Orbitrap': 'LC-ESI-ITFT',
                       'LC-ESI-Q-Exactive Plus': 'LC-ESI-QFT', 'LC-ESI-Q-Exactive Plus Orbitrap Res 14k': 'LC-ESI-QFT', 'LC-ESI-Q-Exactive Plus Orbitrap Res 70k': 'LC-ESI-QFT',
                       'LC-ESI-QQ-LC-ESI-QQ': 'LC-ESI-QQ', 'LC-ESI-QQQ': 'LC-ESI-QQ', 'LC-ESI-QTOF-LC-ESI-QTOF': 'LC-ESI-QTOF', 'LC-ESI-qTOF': 'LC-ESI-QTOF',
                       'LC-ESI-qToF': 'LC-ESI-QTOF', 'LC-ESI-qTof': 'LC-ESI-QTOF', 'LC-ESIMS-qTOF': 'LC-ESI-ITFT', 'N/A-ESI-QFT': 'ESI-QFT', 'N/A-ESI-QTOF': 'ESI-QTOF',
                       'N/A-Linear Ion Trap': 'ESI-IT', 'N/A-N/A': 'ESI-QTOF', 'Negative-Quattro_QQQ:10eV': 'ESI-QQ', 'Negative-Quattro_QQQ:25eV': 'ESI-QQ',
                       'Negative-Quattro_QQQ:40eV': 'ESI-QQ', 'Positive-Quattro_QQQ:10eV': 'ESI-QQ', 'Positive-Quattro_QQQ:25eV': 'ESI-QQ', 'Positive-Quattro_QQQ:40eV': 'ESI-QQ'}
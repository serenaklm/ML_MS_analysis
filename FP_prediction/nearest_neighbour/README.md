This directory contains the code required to reproduce the results for the nearest-neighbour baseline.

The scripts are numbered according to the order in which they should be executed.

### Running the pipeline

1. **Generate DreaMS embeddings**
   ```bash
   python 01_cache_dreaMS_emb.py

This step computes and caches DreaMS embeddings for all datasets and splits.
The embeddings will be saved to the cache directory and reused in later steps.

Run the remaining scripts in numerical order to fully reproduce the nearest-neighbour baseline results.

Each script is designed to be rerun safely: if cached outputs already exist, the computation will be skipped.
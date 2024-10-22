import os
import copy
import yaml
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from utils import read_config
from dataloader import BinnedMSDataset
from modules import MSBinnedModel


def main(args):

    print() 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, default = "./all_configs", help = "Config directory")
    parser.add_argument("--config_file", type = str, default = "base_config.yaml", help = "Config file")
    parser.add_argument("--torch_hub_cache", type = str, default = "./cache", help = "Torch hub cache directory")
    parser.add_argument("--output_dir", type = str, default = "./results", help = "Results output directory")
    parser.add_argument("--debug", action = "store_true", default = False, help = "Set debug mode")
    parser.add_argument("--disable_checkpoint", action = "store_true", default = False, help = "Disable checkpointing")
    parser.add_argument("--wandb", action = "store_true", help = "Enable wandb logging")
    parser.add_argument("--user", type = str, default = "serenakhoolm", help = "Set the user")

    args = parser.parse_args()
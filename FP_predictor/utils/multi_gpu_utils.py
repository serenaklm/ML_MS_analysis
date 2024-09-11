import os 
import torch.distributed as dist

def setup(rank, world_size, port = "02171"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
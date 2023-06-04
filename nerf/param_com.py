import torch
from torch import distributed as dist
from typing import Union
from nerf.mip_model import MipNeRF
from nerf.addtional import ProposalNetwork 

def param_send(model:Union[MipNeRF, ProposalNetwork], dist_ranks: list):
    """ Send to specific machine """
    for param in model.parameters():
        for rank in dist_ranks:
            dist.send(tensor = param.data, dist = rank)
            
def param_recv(model:Union[MipNeRF, ProposalNetwork], source_rank: list):
    """ Receive to specific machine """
    for param in model.parameters():
        dist.send(tensor = param.data, src = source_rank)
        

        
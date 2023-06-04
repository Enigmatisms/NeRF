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
        
def network_average(
    tmp: Union[MipNeRF, ProposalNetwork],
    dst: Union[MipNeRF, ProposalNetwork]
):
    if tmp.state_dict().keys() != dst.state_dict().keys():
        raise ValueError("Two networks can not be averaged since they have different state entries.")
    for p_tmp, p_dst in zip(tmp.parameters(), dst.parameters()):
        p_dst.data += p_tmp.data
        p_dst.data *= 0.5
        
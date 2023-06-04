"""
    Parameter communication
    @author: Qianyue He and other group members
    @date: 2023-6-3
"""

import torch
from torch import distributed as dist
from typing import Union
from nerf.mip_model import MipNeRF
from nerf.addtional import ProposalNetwork 

def param_send(model:Union[MipNeRF, ProposalNetwork], dist_ranks: list):
    """ Send to specific machine """
    for param in model.parameters():
        for rank in dist_ranks:
            dist.send(tensor = param.data, dst = rank)
            
def param_recv(model:Union[MipNeRF, ProposalNetwork], source_rank: list):
    """ Receive to specific machine """
    for param in model.parameters():
        dist.recv(tensor = param.data, src = source_rank)
        
def param_reduce(model:Union[MipNeRF, ProposalNetwork], num_replica: int, dst_rank: int = 0):
    """ Reduce parameter from all other machines """
    for param in model.parameters():
        dist.reduce(tensor = param.data, dst = dst_rank)
        param.data /= num_replica
        
def param_broadcast(model:Union[MipNeRF, ProposalNetwork], src_rank: int = 0):
    """ Broadcast parameter to all other machines """
    for param in model.parameters():
        dist.broadcast(tensor = param.data, src = src_rank)
        
def param_all_reduce(model:Union[MipNeRF, ProposalNetwork], num_replica: int):
    """ All reduce one step model average (each node is a bottleneck) """
    for param in model.parameters():
        dist.all_reduce(tensor = param.data)
        param.data /= num_replica
    
def network_average(
    tmp: Union[MipNeRF, ProposalNetwork],
    dst: Union[MipNeRF, ProposalNetwork]
):
    if tmp.state_dict().keys() != dst.state_dict().keys():
        raise ValueError("Two networks can not be averaged since they have different state entries.")
    for p_tmp, p_dst in zip(tmp.parameters(), dst.parameters()):
        p_dst.data += p_tmp.data
        p_dst.data *= 0.5
        
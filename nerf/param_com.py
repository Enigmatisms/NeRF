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

def param_send(model:Union[MipNeRF, ProposalNetwork], dist_ranks: list, group = None):
    """ Send to specific machine """
    for param in model.parameters():
        for rank in dist_ranks:
            dist.send(tensor = param.data, dst = rank, group = group)
            
def param_recv(model:Union[MipNeRF, ProposalNetwork], source_rank: list, group = None):
    """ Receive to specific machine """
    for param in model.parameters():
        dist.recv(tensor = param.data, src = source_rank, group = group)
        
def param_recv_avg(
    model: Union[MipNeRF, ProposalNetwork],
    tmp: Union[MipNeRF, ProposalNetwork],
    weights: list, source_ranks: list, self_rank: int = 0, group = None
):
    """ Receive parameters and do the weighted average """
    for p_tmp, p_model in zip(tmp.parameters(), model.parameters()):
        p_model.data *= weights[self_rank]
        for src_rank in source_ranks:
            dist.recv(tensor = p_tmp.data, src = src_rank, group = group, group = group)
            p_model.data += weights[src_rank] * p_tmp.data
        
def param_reduce(model:Union[MipNeRF, ProposalNetwork], weights: list, self_rank: int, dst_rank: int = 0, group = None):
    """ Reduce parameter from all other machines 
        Before sending: multiply the param by the weight
    """
    for param in model.parameters():
        param.data *= weights[self_rank]
        dist.reduce(tensor = param.data, dst = dst_rank, group = group)
        
def param_broadcast(model:Union[MipNeRF, ProposalNetwork], src_rank: int = 0, group = None):
    """ Broadcast parameter to all other machines """
    for param in model.parameters():
        dist.broadcast(tensor = param.data, src = src_rank, group = group)
        
def param_all_reduce(model:Union[MipNeRF, ProposalNetwork], group = None):
    """ All reduce one step model average (each node is a bottleneck) 
        Before calling this function, all the parameters are weighted
    """
    for param in model.parameters():
        dist.all_reduce(tensor = param.data, group = group)
        
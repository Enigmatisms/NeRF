""" Shuffling the local part of dataset
    This might need to use DDP
    @author: Qianyue He
    @date: 2023-5-31
"""

import random
from copy import deepcopy
import torch.distributed as dist
from typing import Optional, List
from torchvision import transforms
from dataset import CustomDataSet, AdaptiveResize
from torch.utils.data import DistributedSampler, Dataset, DataLoader

class LocalShuffleSampler(DistributedSampler):
    """ Distributed sampler does not satisfy my requirement
        Since model average requires that each model accesses and uses its own data
        dist_sampler's shuffling mechanism will produce indices
        that leads to the access of other parts of the dataset where
        one specific model should not

        Note that now, each model can have different work load, we take the minimum
        number of samples

        indices: 2D list, shape (number of models, <dynamic> number of samples)
    """
    def __init__(self, dataset: Dataset, indices: List[List[int]],
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0) -> None:
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        num_replicas = len(indices)
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas

        # get the minimum of all samples
        self.min_sample = min([len(index_list) for index_list in indices])

        self.rank = rank
        self.epoch = 0
        self.drop_last = False

        self.samples = deepcopy(indices)

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = deepcopy(self.samples[self.rank])
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            random.seed(self.seed + self.epoch)         # This might be redundant
            random.shuffle(indices)
            indices = indices[:self.min_sample]         # make sure all machines have same number of samples
            # TODO: the requirement that all machines having the same number of samples can be LIFTED
            # This can spark a new strategy (and a new situation)
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        print(indices)
        return iter(indices)
    
    def __len__(self):
        """ If we do not cut off at the min sample, this should be changed """
        return self.min_sample
    
    """ Note that set_epoch is inherited """

if __name__ == "__main__":
    # Simple unit test
    transform_funcs = transforms.Compose([
        AdaptiveResize(0.5),
        transforms.ToTensor(),
    ])


    trainset = CustomDataSet(f"../../dataset/nerf_synthetic/lego/", transform_funcs, 
        1.0, True, use_alpha = False, white_bkg = True)
    sampler = LocalShuffleSampler(trainset, [[0, 6, 7, 5, 2]], rank=0)
    dataloader = DataLoader(trainset, batch_size = 1, sampler = sampler, shuffle = False)

    for train_img, train_tf in dataloader:
        print(train_img.shape, train_tf.shape)
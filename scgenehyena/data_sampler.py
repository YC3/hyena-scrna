import torch
from typing import List, Dict, Union, Optional, Sequence, Iterable
import numpy as np
from numpy import ndarray

import tiledb
import tiledbsoma as soma
import cellxgene_census
import inspect
import importlib

from torch.utils.data import Dataset
from torch.utils.data import Sampler, SubsetRandomSampler, BatchSampler



# data samplers
class IdentitySampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)



class SubsetRandomBatchSampler(Sampler[List[int]]):
    
    r"""Shuffle input indices randomly, and create batches according to ``batch size``.

    Arguments:
        indice: A list of indice.
        batch_size: Size of mini-batch.
        shuffle: If ``True``, the sampler will shuffle the indice before split into batches.
        drop_last: If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size``.
    """

    def __init__(
        self,
        indice: Union[Sequence[int]],
        batch_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.indice = indice
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if shuffle:
            self.indice_sampler = SubsetRandomSampler(self.indice)
        else:
            self.indice_sampler = IdentitySampler(self.indice)

        self.batch_samplers = BatchSampler(self.indice_sampler, batch_size, drop_last)

        if shuffle:
            # maintain a mapping from sample batch index to batch sampler
            _id_to_batch_sampler = []
            for i, batch_sampler in enumerate(self.batch_samplers):
                _id_to_batch_sampler.extend([i] * len(batch_sampler))
            self._id_to_batch_sampler = np.array(_id_to_batch_sampler)

            assert len(self._id_to_batch_sampler) == len(self)

            self.batch_sampler_iterrators = [
                batch_sampler.__iter__() for batch_sampler in self.batch_samplers
            ]

    def __iter__(self) -> Iterable[List[int]]:
        for batch_sampler in self.batch_samplers:
            yield batch_sampler

    def __len__(self) -> int:
        return sum(len(batch_sampler) for batch_sampler in self.batch_samplers)

from torch.utils.data import Sampler
from typing import List
import numpy as np



class StratifiedCellSampler(Sampler[int]):

    def __init__(
        self,
        cell_types: List[str],
        samples_per_epoch: int,
    ):

        """
        Produces indices with inverse-frequency stratification.
        Args:
            cell_types: a list of cell type labels aligned one-to-one with the dataset index space
            samples_per_epoch: number of cells get sampled per epoch
        """

        self.cell_types = np.array(cell_types)
        self.samples_per_epoch = samples_per_epoch

        # calculate the sampling probability
        unique, counts = np.unique(cell_types, return_counts=True)
        weights = counts.sum()/(counts*len(unique))
        
        self.prob = weights/weights.sum()
        self.type_to_indices = {ct:np.where(self.cell_types == ct)[0] for ct in unique}

    def __iter__(self):
        for _ in range(self.samples_per_epoch):
            # choose a cell type
            ct = np.random.choice(list(self.type_to_indices.keys()), p=self.prob)
            # choose a cell id
            yield int(np.random.choice(self.type_to_indices[ct]))

    def __len__(self):
        return self.samples_per_epoch

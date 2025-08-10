
import os
import random
from typing import List, Dict, Union, Optional, Sequence, Iterable

import numpy as np
from numpy import ndarray
import pandas as pd

import tiledb
import tiledbsoma as soma
import cellxgene_census
import inspect
import importlib

import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler, SubsetRandomSampler, BatchSampler

import scanpy as sc
import h5py
from anndata._io.specs import read_elem
from scgenehyena.utils import get_unique_counts
from collections import defaultdict



class StratifiedCellSampler(Dataset):
    
    def __init__(
        self, 
        h5ad_files, 
        genes,
        cell_type_key="cell_type", 
        samples_per_epoch=1000
    ):
        """
        On-the-fly stratified sampler for scRNA-seq foundation model pretraining.
        Args:
            h5ad_files: List of paths to H5AD files
            cell_type_key: Column name in adata.obs containing cell type labels
            samples_per_epoch: Virtual dataset size per epoch
        """
        
        self.h5ad_files = h5ad_files
        self.genes = genes
        self.cell_type_key = cell_type_key
        self.samples_per_epoch = samples_per_epoch
        
        self.file_index = self._build_file_index()
        self.cell_type_weights = self._compute_stratification_weights()
        
        self.worker_file_cache = {}

    
    def _build_file_index(self):
        
        """
        Create lightweight index of cell type distributions per file
        """
        
        file_index = {}
        for path in self.h5ad_files:    
            with h5py.File(path) as f:
                obs = read_elem(f["obs"])

                cell_types = obs[self.cell_type_key].values
                file_index[path] = get_unique_counts(cell_types)
     
        return file_index
        

    def _compute_stratification_weights(self):
        
        """
        Compute inverse frequency weights for cell types
        """
        
        global_type_counts = defaultdict(int)
        
        for meta in self.file_index.values():
            for cell, count in meta.items():
                global_type_counts[cell] += count
                
        total_cells = sum(global_type_counts.values())
        
        return {cell: total_cells / (count * len(global_type_counts))
                for cell, count in global_type_counts.items()
               }

    def _get_worker_cache(self):
        
        """
        Get worker-specific file cache
        """
        
        worker_id = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0
        
        if worker_id not in self.worker_file_cache:
            self.worker_file_cache[worker_id] = {}
            
        return self.worker_file_cache[worker_id]

    def _sample_cell_type(self):
        
        """
        Sample cell type based on stratification weights
        """
        
        types, weights = zip(*self.cell_type_weights.items())
        probs = np.array(weights) / sum(weights)
        
        return np.random.choice(types, p=probs)

    def _sample_file(self, cell_type):
        
        """
        Sample file containing the cell type
        """
        
        valid_files = [
            path for path, meta in self.file_index.items() 
            if cell_type in meta.keys()
        ]
        
        return random.choice(valid_files)

    def _sample_cell(self, file_path, cell_type):
        
        """
        Sample a specific cell from file
        """
        
        cache = self._get_worker_cache()
        
        if file_path not in cache:
            cache[file_path] = {
                'adata': sc.read_h5ad(file_path, backed='r'),
                'cell_indices': defaultdict(list)
            }

            ct_series = cache[file_path]['adata'].obs[self.cell_type_key]
            for idx, ct in enumerate(ct_series):
                cache[file_path]['cell_indices'][ct].append(idx)
                
        cell_idx = random.choice(cache[file_path]['cell_indices'][cell_type])
        
        expr = pd.Series(cache[file_path]['adata'][cell_idx].X.toarray().flatten(),
                         cache[file_path]['adata'].var.index.tolist()
                         )
        expr = expr.reindex(self.genes).fillna(0.0).tolist()

        return np.array(expr, copy=True)
        

    def __len__(self):
        
        return int(self.samples_per_epoch)

    def __getitem__(self, idx):
        
        # 1. sample cell type with stratification
        cell_type = self._sample_cell_type()
        
        # 2. sample file containing this cell type
        file_path = self._sample_file(cell_type)
        
        # 3. sample and load specific cell
        expr = self._sample_cell(file_path, cell_type)
        
        return {'expression': torch.tensor(expr.copy(), dtype=torch.float32),
                'cell_type': cell_type
                }



# data samplers
class IdentitySampler(Sampler):
    
    def __init__(self, indices: Sequence[int]):
        """
        Samples elements sequentially from a given list of indices, without replacement.
        Args:
            indices (sequence): a sequence of indices
        """
        self.indices = indices

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)



class SubsetRandomBatchSampler(Sampler[List[int]]):

    def __init__(
        self,
        indice: Union[Sequence[int]],
        batch_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Shuffle input indices randomly, and create batches according to batch size.
        Args:
            indice: A list of indice.
            batch_size: Size of mini-batch.
            shuffle: If ``True``, the sampler will shuffle the indice before split into batches.
            drop_last: If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size``.
        """
        
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

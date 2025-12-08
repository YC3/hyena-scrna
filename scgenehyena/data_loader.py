
import os
import random
from typing import Any, List, Dict, Union, Optional, Sequence, Iterable

import numpy as np
from numpy import ndarray
import pandas as pd

import tiledb
import tiledbsoma as soma
import cellxgene_census
import inspect
import importlib
import random
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler, SubsetRandomSampler, BatchSampler

from scgpt.tokenizer.gene_tokenizer import GeneVocab

import anndata
import h5py
from anndata._io.specs import read_elem
from collections import defaultdict

from scgenehyena.utils import get_unique_counts
from scgenehyena.scgpt_gene_tokenizer import tokenize_and_pad_batch



## TODO: random mask
## TODO: add vocab, and convert tokens to vocab's

class StratifiedVeloDataset(Dataset):
    
    def __init__(
        self, 
        h5ad_files: List[str], 
        genes: List[str],
        cell_type_key: str = 'cell_type', 
        batch_key: str = None,
        samples_per_epoch: int = 1000,
        max_len: int = 2000,
        s_key: str = 'spliced',
        u_key: str = 'unspliced',
        vocab: GeneVocab = None,
        pad_token: str = "<pad>",
        pad_value: str = "-2",
        append_cls: bool = True,
        include_zero_gene: bool = True,
        cls_token: str = "<cls>",
        return_pt: bool = True,
    ):

        """
        On-the-fly stratified sampler for scRNA-seq foundation model pretraining.
        Args:
            h5ad_files: list of paths to h5ad files (with layers["spliced"], layers["unspliced"])
            genes: global gene list to align to (same order your model/vocab expects)
            cell_type_key: column name in adata.obs containing cell type labels
            batch_key: column name in adata.obs containing batch information
            samples_per_epoch: virtual dataset size per epoch
            max_len: maximum sequence length
            s_key: column name in adata.layers containing spliced read counts
            u_key: column name in adata.layers containing unspliced read counts
        """
        
        self.h5ad_files = h5ad_files
        self.genes = genes
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key
        self.samples_per_epoch = samples_per_epoch
        self.max_len = max_len
        self.s_key = s_key
        self.u_key = u_key
        self.vocab = vocab
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.append_cls = append_cls
        self.include_zero_gene = include_zero_gene
        self.cls_token = cls_token
        self.return_pt = return_pt
        
        self.file_index = self._build_file_index()
        
        # cell type encoding
        all_cell_types = sorted({c for meta in self.file_index.values() for c in meta})
        self.c2i = {c: i for i, c in enumerate(all_cell_types)}
        self.i2c = {i: c for c, i in self.c2i.items()}

        # batch encoding
        if self.batch_key is not None:
            all_batches = self._scan_all_batches()
            self.b2d = {b: i for i, b in enumerate(sorted(all_batches))}
            self.i2b = {i: b for b, i in self.b2d.items()}
        else:
            self.b2d = None
            self.i2b = None

        self.cell_type_weights = self._compute_stratification_weights()
        
        self.worker_file_cache = {}


          
    
    def _build_file_index(self):
        
        """
        Create index of cell type distributions per file
        """
        
        file_index = {}
        for path in self.h5ad_files:    
            with h5py.File(path, 'r') as f:
                obs = read_elem(f['obs'])

                cell_types = obs[self.cell_type_key].values
                file_index[path] = get_unique_counts(cell_types)
     
        return file_index
        

    def _scan_all_batches(self):

        """
        Collect the global set of batch labels across all files
        """

        batches = set()
        for path in self.h5ad_files:
            with h5py.File(path, "r") as f:
                obs = read_elem(f["obs"])
                
                if self.batch_key in obs.columns:
                    batches.update(map(str, obs[self.batch_key].values))
        return batches


    def _compute_stratification_weights(self):
        
        """
        Compute inverse frequency weights for cell types
        """
        
        global_type_counts = defaultdict(int)
        
        for meta in self.file_index.values():
            for cell, count in meta.items():
                global_type_counts[cell] += count
                
        total_cells = sum(global_type_counts.values())
        n_types = len(global_type_counts)
        
        return {
            cell: total_cells/(count*n_types)
            for cell, count in global_type_counts.items()
            }


    def _sample_cell_type(self):
        
        """
        Sample cell type based on stratification weights
        """
        
        types, weights = zip(*self.cell_type_weights.items())
        probs = np.array(weights)/sum(weights)
        
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


    def _sample_cell(
        self, 
        file_path, 
        cell_type
    ):
        
        """
        Sample a specific cell from file
        """
        
        cache = self._get_worker_cache()
        
        if file_path not in cache:

            adata = anndata.read_h5ad(file_path, backed="r")
            # preprocessing here
            # do preprocessing here: Apply minimal, standardized QC & formatting
            # don't do heavy biological normalization (log1p, scaling, HVG) before pretraining

            ct_series = adata.obs[self.cell_type_key]

            cell_indices = defaultdict(list)
            for i, ct in enumerate(ct_series):
                cell_indices[ct].append(i)

            cache[file_path] = {
                "adata": adata, 
                "cell_indices": cell_indices,
            }

        adata = cache[file_path]["adata"]
        cell_idx = random.choice(cache[file_path]["cell_indices"][cell_type])

        # extract spliced / unspliced for this cell
        s = adata.layers["spliced"][cell_idx].toarray().flatten()
        u = adata.layers["unspliced"][cell_idx].toarray().flatten()
        gene_names = adata.var_names
    
        # align to global gene set
        map_s = dict(zip(gene_names, s))
        map_u = dict(zip(gene_names, u))
        exp_s = np.fromiter((map_s.get(g, 0.0) for g in self.genes), float)
        exp_u = np.fromiter((map_u.get(g, 0.0) for g in self.genes), float)

        # batch info
        batch = adata.obs[self.batch_key][cell_idx]

        return (
            torch.tensor(exp_s, dtype=torch.float32), 
            torch.tensor(exp_u, dtype=torch.float32),
            batch
            #, cell_idx
        )
    

    def _get_worker_cache(self):
        
        """
        Get worker-specific file cache
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        if worker_id not in self.worker_file_cache:
            self.worker_file_cache[worker_id] = {}
            
        return self.worker_file_cache[worker_id]
        

        
    def __len__(self):
        
        return int(self.samples_per_epoch)


    def __getitem__(self, idx): ## TODO: check idx
        
        # 1. sample cell type with stratification
        cell_type = self._sample_cell_type()
        
        # 2. sample file containing this cell type
        file_path = self._sample_file(cell_type)
        
        # 3. sample and load specific cell
        exp_s, exp_u, batch = self._sample_cell(file_path, cell_type)

        return {
                's':exp_s.reshape(-1, exp_s.shape[0]), 
                'u':exp_u.reshape(-1, exp_u.shape[0]), 
            }
        # 4. tokenization
        tokenized_data  = tokenize_and_pad_batch(
            data_dict={
                's':exp_s.reshape(-1, exp_s.shape[0]), 
                'u':exp_u.reshape(-1, exp_u.shape[0]), 
            },
            genes=self.genes,
            max_len=self.max_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=self.append_cls,
            include_zero_gene=self.include_zero_gene, 
            cls_token=self.cls_token, 
            return_pt=self.return_pt,
        )
        
        tokenized_data['expr_mask'] = (tokenized_data['values_s'] > 0) | (tokenized_data['values_u'] > 0)
        tokenized_data['cell_type'] = cell_type
        tokenized_data['batch'] = batch

        return tokenized_data



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

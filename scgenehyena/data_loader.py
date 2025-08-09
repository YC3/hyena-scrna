import torch
from typing import List, Dict, Union, Optional, Sequence, Iterable
import numpy as np
from numpy import ndarray

from datasets import Dataset
import cellxgene_census

# TODO: add other loss functions for other training objectives

def save(dataset: Union[Dataset, ndarray] = None, 
         out_format: Union['json', 'parquet'] = 'json', 
         out_path: Optional[str] = None, 
         compression: Optional[str] = 'gzip'):
    
    if out_format == 'json':
        df.to_json(path_or_buf = out_path, compression = compression)
    elif out_format == 'parquet':
        df.to_parquet(path = out_path, compression = compression)


# getting data from cellxgene
def create_data_generator(random_idx):
  for idx in random_idx:
      yield idx




def get_batch_adata(census, data_generator, norm_method = None):



    
    # query tiledb
    adata = cellxgene_census.get_anndata(
        census, 
        "Mus musculus", 
        obs_coords=next(data_generator),
        #obs_value_filter = f'tissue == "brain"'
    )

    return adata # this way, makes preprocessing more flexible
    
    np.random.seed(3)
    data_pt = {
    'gene_ids': torch.tensor(np.tile(np.random.randint(0, 52417, 52417), (3, 1))),
    'gene_values':torch.Tensor(adata.X.todense())}

    return data_pt















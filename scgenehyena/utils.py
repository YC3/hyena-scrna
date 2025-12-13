import torch
import anndata
from typing import List, Dict
import numpy as np



def get_toy_data(
    num_genes: int = 20000,
    dim: int = 4,
    mask_ratio: float = 0.0,
    seed: int = 1
):
    
    input_data = torch.randn(dim, num_genes, 1)

    if mask_ratio > 0.0:
        mask = torch.rand(dim, num_genes, 1) > mask_ratio
        masked_input = input_data*mask.float()

        return masked_input, mask
    else:
        return input_data



def get_unique_counts(data: List):
    
    unique, counts = np.unique(data, return_counts=True)
    
    return dict(zip(unique, counts))



def build_cell_index(
    h5ad_files: List[str], 
    cell_type_key: str = None,
) -> Dict[int, str]:

    cell_type_map = {}
    cell_index = []
    for path in h5ad_files:

        adata = anndata.read_h5ad(path, backed="r")

        for i, ct in enumerate(adata.obs[cell_type_key]):
            cell_index.append((path, i))
            cell_type_map[len(cell_index) - 1] = ct 

    return cell_type_map, cell_index



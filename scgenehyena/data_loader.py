import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import anndata



_WORKER_ADATA_CACHE = {}

class VeloCellDataset(Dataset):

    def __init__(
        self,
        cell_index: List[Tuple[str, int]],
        cell_type_map: Dict[int, str],
        t_key: str,
        s_key: str,
        u_key: str,
        batch_key: Optional[str],
    ):

        """
        Given an integer index i, return the raw data for exactly one cell.
        Args:
            cell_idnex: a list of tuples that map global cell indice to file paths, constructed by the function build_cell_index()
            cell_type_map: a dictionary that maps global cell indice to cellt ype, another output of the function build_cell_index()
            batch_key: column name in adata.obs containing batch information
            t_key: the key name in adata.layers containing the total read counts
            s_key: the key name in adata.layers containing the spliced read counts
            u_key: the key name in adata.layers containing the unspliced read counts
        """

        self.cell_index = cell_index
        self.cell_type_map = cell_type_map
        self.batch_key = batch_key
        self.t_key = t_key
        self.s_key = s_key
        self.u_key = u_key


    def _get_adata(self, path: str):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0

        if wid not in _WORKER_ADATA_CACHE:
            _WORKER_ADATA_CACHE[wid] = {}

        worker_cache = _WORKER_ADATA_CACHE[wid]

        if path not in worker_cache:
            worker_cache[path] = anndata.read_h5ad(path, backed="r")

        return worker_cache[path]

    def __len__(self):

        return len(self.cell_index)


    def __getitem__(self, idx):
        
        path, cell_idx = self.cell_index[idx]
        
        adata = self._get_adata(path)

        t = adata.layers[self.t_key][cell_idx]
        s = adata.layers[self.s_key][cell_idx]
        u = adata.layers[self.u_key][cell_idx]

        cell_type = self.cell_type_map[cell_idx]
        batch = (
            adata.obs[self.batch_key][cell_idx]
            if self.batch_key else None
        )

        return {
            "t": torch.tensor(t),
            "s": torch.tensor(s),
            "u": torch.tensor(u),
            "cell_type": cell_type,
            "batch": batch,
        }



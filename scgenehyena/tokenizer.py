from typing import List, Dict, Union, Tuple
import numpy as np
import torch
from torchtext.vocab import Vocab

from .scgpt_gene_tokenizer import GeneVocab



class VeloTokenizer:

    def __init__(
        self,
        genes: List[str],
        vocab: GeneVocab = None,
        max_len: int = 2000,
        pad_token: str = "<pad>",
        pad_value: str = -2,
        append_cls: bool = True,
        include_zero_gene: bool = False,
        cls_token: str = "<cls>",
    ):

        """
        Convert raw per-cell expression vectors into model-ready token sequences.
        Args:
            genes: global gene list to align to (same order your model/vocab expects)
            vocab: maps gene names and special tokens to integer IDs
            max_len: the maximum sequence length after tokenization
            pad_token: the token used to pad sequences shorter than max_len
            pad_value: the numerical value used to pad expression tensors
            append_cls: whether to prepend a <cls> token to the sequence
            include_zero_gene: controls whether genes with zero total expression are included
            cls_token: the CLS token, defaults to `<cls>`
        """

        self.genes = genes
        self.vocab = vocab
        self.max_len = max_len
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.append_cls = append_cls
        self.include_zero_gene = include_zero_gene
        self.cls_token = cls_token

    def __call__(self, sample):
        tokenized = tokenize_and_pad_batch(
            data_dict={
                "t": sample["t"].unsqueeze(0),
                "s": sample["s"].unsqueeze(0),
                "u": sample["u"].unsqueeze(0),
            },
            genes=self.genes,
            max_len=self.max_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=self.append_cls,
            include_zero_gene=self.include_zero_gene,
            cls_token=self.cls_token,
            return_pt=True,
        )

        tokenized["expr_mask"] = (tokenized["values_s"] > 0) | (tokenized["values_u"] > 0)
        tokenized["cell_type"] = sample["cell_type"]
        tokenized["batch"] = sample["batch"]

        return tokenized




def tokenize_batch(
    data: Dict[str, torch.Tensor],
    genes: List[str],
    vocab: GeneVocab,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>",
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).
    Modified from scGPT (https://github.com/bowang-lab/scGPT).
    
    Args:
        data (np.array): A list of a batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (tensor-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_id, count) of non zero gene expressions.
    """  

    # convert gene names to ids
    gene_ids = torch.Tensor([vocab[x] for x in genes])

    tokenized_data = {}
    for key in data.keys():
        if data[key].shape[1] != len(gene_ids):
            raise ValueError(
                f"Number of features in data ({data[key].shape[1]}) does not match "
                f"number of gene_ids ({len(gene_ids)})."
            )
        if mod_type is not None and data[key].shape[1] != len(mod_type):
            raise ValueError(
                f"Number of features in data ({data[key].shape[1]}) does not match "
                f"number of mod_type ({len(mod_type)})."
            )

        _ = []
        for i in range(len(data[key])):
            row = data[key][i]
            mod_types = None
            if include_zero_gene:
                values = row
                genes = gene_ids
                if mod_type is not None:
                    mod_types = mod_type
            else:
                if key=='t':
                    idx = np.nonzero(row)
                values = row[idx]
                genes = gene_ids[idx]
                if mod_type is not None:
                    mod_types = mod_type[idx]
            if append_cls:
                genes = np.insert(genes, 0, cls_id)
                values = np.insert(values, 0, 0)

                if mod_type is not None:
                    mod_types = np.insert(mod_types, 0, cls_id_mod_type)
            if return_pt:
                genes = genes.long()
                values = values.float()
                if mod_type is not None:
                    mod_types = torch.from_numpy(mod_types).long()
            _.append((genes, values, mod_types))
        
        tokenized_data[key] = _
        
    return tokenized_data



def pad_batch(
    batch:  Dict[str, List[Tuple]],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:

    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].
    Modified from scGPT (https://github.com/bowang-lab/scGPT).

    Args:
        batch (list): A list of tuple (gene_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of gene_id and count.
    """

    keys = list(batch.keys())

    batch_size = len(batch[keys[0]])
    if batch_size > 1:
        max_ori_len = max(len(batch[keys[0]][i][0]) for i in range(batch_size))
        max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
    gene_ids_list = []
    values_dict = {k:[] for k in keys}
    mod_types_list = []

    for i in range(batch_size):
        gene_ids, _, mod_types = batch[keys[0]][i]
        values_d = {}

        if len(gene_ids) > max_len:
            # sample max_len genes
            # once a list of genes is sampled for one input, use the same idx for sampling other inputs
            if not cls_appended:
                idx = np.random.choice(len(gene_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids = gene_ids[idx]
            for k in keys:
                _, values, _ = batch[k][i]
                values_d[k] = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]
        else:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        size=(max_len - len(gene_ids),), 
                        fill_value=pad_id, 
                        dtype=gene_ids.dtype
                    ),
                ]
            )
            for k in keys:
                _, values, _ = batch[k][i]
                values_d[k] = torch.cat(
                    [
                        values,
                        torch.full(
                            size=(max_len - len(values),), 
                            fill_value=pad_value, 
                            dtype=values.dtype
                        ),
                    ]
                )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            size=(max_len - len(mod_types),),
                            fill_value=mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )

        gene_ids_list.append(gene_ids)
        for k in keys:
            values_dict[k].append(values_d[k])
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "gene_ids": torch.stack(gene_ids_list, dim=0),
    }
    for k in keys:
        batch_padded['values_'+k] = torch.stack(values_dict[k], dim=0)

    # not modified
    if mod_types is not None:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
    return batch_padded



def tokenize_and_pad_batch(
    data_dict: Dict[str, torch.Tensor],
    genes: List[str],
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:

    """
    Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
    Modified from scGPT (https://github.com/bowang-lab/scGPT).
    """

    cls_id = vocab[cls_token]
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]
    tokenized_data = tokenize_batch(
        data_dict,
        genes,
        vocab,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
    )

    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
    )

    return {k:batch_padded[k].squeeze() for k in batch_padded.keys()}


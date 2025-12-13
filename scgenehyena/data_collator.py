
from typing import List, Dict, Any
import torch



class TokenizeAndCollate:

    def __init__(
        self, 
        tokenizer, 
        collator
    ):
    
        self.tokenizer = tokenizer
        self.collator = collator

    def __call__(self, batch):
        # tokenize per-sample
        tokenized = [self.tokenizer(sample) for sample in batch]
        # batch + MLM + masks
        return self.collator(tokenized)


class VeloDataCollator:
    
    def __init__(
        self,
        pad_value: int = -2,
        mlm_probability: float = 0.3,
        mask_value: int = -1,
        keep_first_n_tokens: int = 1,
        keys2mask: List[str] = ("values_t", "values_s", "values_u"),
        use_attention_mask: bool = True,
    ):
    
        """
        Stasking data in a batch, adding attention and mlm masks.
        Args:
            pad_value: the numerical value used to pad expression tensors
            mlm_probability: the proportions of data to be masked for MLM
            mask_value: the mask value for MLM
            keep_first_n_tokens: protects <cls> token from mlm masking
            keys2mask: which data slice to be masked
            use_attention_mask: whether to create the attention mask
        """

        self.pad_value = pad_value
        self.mlm_probability = mlm_probability
        self.mask_value = mask_value
        self.keep_first_n_tokens = keep_first_n_tokens
        self.keys2mask = list(keys2mask)
        self.use_attention_mask = use_attention_mask


    def __call__(
        self, 
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        out = {}
        for key in batch[0].keys():

            values = [sample[key] for sample in batch]
            if torch.is_tensor(values[0]):
                out[key] = torch.stack(values, dim=0)
            else:
                out[key] = values

        # attention mask
        if self.use_attention_mask:
            out["attention_mask"] = out[self.keys2mask[0]] != self.pad_value

        # mlm mask
        out = mlm_mask(
            data=out,
            mlm_probability=self.mlm_probability,
            pad_value=self.pad_value,
            mask_value=self.mask_value,
            keep_first_n_tokens=self.keep_first_n_tokens,
            keys2mask=self.keys2mask,
        )

        return out



def mlm_mask(
    data: Dict[str, torch.Tensor],
    keys2mask: List[str],
    mlm_probability: float = 0.3,
    pad_value: int = -2,
    mask_value: int = -1,
    keep_first_n_tokens: int = 1,
) -> torch.Tensor:
    
    """
    Mask the expression values with MLM.
    """
    
    device = data[keys2mask[0]].device
    shape = data[keys2mask[0]].shape

    probability_matrix = torch.full(shape, mlm_probability)
    
    # set padded postion probability to 0
    probability_matrix[data[keys2mask[0]].eq(pad_value)] = 0
    if keep_first_n_tokens > 0:
        probability_matrix[:, : keep_first_n_tokens] = 0

    mask = torch.bernoulli(probability_matrix).bool()
    mask = mask.to(device)

    data_ = data.copy()
    for key in keys2mask:
        data_['masked_'+key.split('_')[-1]] = data_[key].masked_fill(mask, mask_value)
    data_['mlm_mask'] = mask
                          
    return data_
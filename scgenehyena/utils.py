import torch


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
    
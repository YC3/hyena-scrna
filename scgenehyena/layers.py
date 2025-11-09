import torch
import torch.nn as nn

from src.models.sequence.hyena import HyenaOperator # hyena-dna (https://github.com/HazyResearch/hyena-dna)



class GeneEmbedding(nn.Module):
    
    def __init__(
        self, 
        dim: int = 512,
    ):
        
        super().__init__()
        # Projects scalar expression → high-dim vector
        self.value_proj = nn.Sequential(
            nn.Linear(1, dim),  # Input: (batch, genes, 1)
            nn.GELU(),
            nn.LayerNorm(dim)
        )
        
    def forward(
        self, 
        x: torch.Tensor
    ):
        """x: (batch, num_genes, 1) tensor of expression values"""
        
        return self.value_proj(x)  # → (batch, num_genes, dim)



class PositionEmbedding(nn.Module):
    
    def __init__(
        self, 
        max_genes: int = 20000, 
        dim: int = 512,
    ):
        
        super().__init__()
        # Each gene index gets a unique vector
        self.pos_embed = nn.Parameter(torch.randn(1, max_genes, dim))
        
    def forward(
        self, 
        x: torch.Tensor
    ):
        """x: embedded genes (batch, num_genes, dim)"""

        return x + self.pos_embed[:, :x.size(1), :]



class ConvAfterHyena(nn.Module):
    
    def __init__(
        self, d_model: int, 
        conv_kernel_size: int = 3,
    ):

        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            padding="same", 
            groups=d_model,
            bias=False,
        )
        self.activation = nn.GELU()

    def forward(
        self, 
        x: torch.Tensor,
    ):
        """x: input tensor （batch, seq_len, dim)"""

        x = x.permute(0, 2, 1) # Conv1D expects (batch, channels, seq_len)
        x = self.conv(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)
        return x



class HyenaBlock(nn.Module):
    
    def __init__(
        self, 
        d_model: int, 
        l_max: int = 32768, 
        order: int = 2,
        filter_order: int = 64,
        dropout: float = 0.1,
        filter_dropout: float = 0.1,
        num_heads: int = 1,
        depth: int = 12,
        conv_kernel_size: int = 3,
        hyena_dropout: float = 0.1,
    ):

        """
        HyenaBlock: stacked Hyena + depthwise-conv blocks

        Each block:
            x -> LayerNorm
              -> HyenaOperator
              -> depthwise Conv1D (ConvAfterHyena)
              -> GELU
              -> Dropout
            + residual

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max (int): Maximum input sequence length. Defaults to 32768
            order (int): Depth of the Hyena recurrence. Defaults to 2
            filter_order (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            num_heads (int): Number of heads. Defaults to 1
            depth (int): Number of the Hyena layers
            conv_kernel_size (int): Kernal size of the convolution step 
            hyena_dropout (float): Dropout after Hyena+Conv in each block.
        """
        
        super().__init__()

        hyena = HyenaOperator(
            d_model=d_model,
            l_max=l_max,
            order=order,
            filter_order=filter_order,
            dropout=dropout,
            filter_dropout=filter_dropout,
            num_heads=num_heads,
        )

        conv = ConvAfterHyena(
            d_model=d_model, 
            conv_kernel_size=conv_kernel_size
        )

        hyena_block = nn.Sequential(
            nn.LayerNorm(d_model),
            hyena,
            conv,
            nn.GELU(),
            nn.Dropout(hyena_dropout)
        )

        self.hyena_blocks = nn.ModuleList(
            [hyena_block for _ in range(depth)]
        )


    def forward(
        self, 
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
    ):

        """x: input tensor (batch, seq_len, dim)"""

        residual = x
        for layer in self.hyena_blocks:
            x = layer(x) + residual
            residual = x

        if src_key_padding_mask if not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
            
        return x






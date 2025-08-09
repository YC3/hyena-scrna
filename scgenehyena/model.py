import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GeneEmbedding, PositionEmbedding, Hyena

from einops import rearrange, repeat



class ScGeneHyena(nn.Module):
    
    def __init__(self,
        num_genes: int = 20000,
        dim: int = 512,
        depth: int = 12,
        max_length: int = 32768,
        order: int = 2,
        num_heads: int = 1,
        conv_kernel_size: int = 3,
        emb_dropout: float = 0.0,
        hyena_dropout: float = 0.1,
        pretrain_head: str = 'masked_reconstruction',
        num_blocks: int = 1,
    ):
        r"""
        ScGeneHyena: Hyena-based foundation model for single-cell genomics
        Args:
            num_genes (int): Number of the genes in the input data
            dim (int): Dimension of the input and output embeddings (width of the layer)
            depth (int): Number of the Hyena layers
            max_length (int): Maximum input sequence length. Defaults to 32768
            order (int): Depth of the Hyena recurrence. Defaults to 2
            num_heads (int): Number of heads. Defaults to 1
            conv_kernel_size (int): Kernal size of the convolution step
            emb_dropout (int): Dropout probability. Defaults to 0.0
            hyena_dropout (int): Dropout probability after HyenaOperation. Defaults to 0.1
            pretrain_head (int): 'masked' or 'contrastive'
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
        """
        
        super().__init__()
        self.num_genes = num_genes
        self.pretrain_head = pretrain_head
        
        # Gene embedding 
        self.gene_embedding = GeneEmbedding(dim)
        
        # Positional embedding
        self.pos_embedding = PositionEmbedding(max_length, dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        # Hyena layers
        self.hyena_layers = Hyena(
            d_model = dim, 
            l_max = max_length, 
            order = order,
            num_heads = num_heads,
            conv_kernel_size = 3,
            depth = depth,
            hyena_dropout = hyena_dropout,
        )
        
        # Pretraining objectives
        if pretrain_head == 'masked_reconstruction':
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )
        elif pretrain_head == 'contrastive':
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 256),
                nn.BatchNorm1d(256)
            )
        
        # Biological context projection
        self.context_proj = nn.Linear(dim, 4)

    
    def forward(self, x, mask=None):
        
        """
        Input: (batch, genes, 1) normalized expression tensor
        """
         
        # Embed genes with positional information
        x = self.gene_embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        # Process through Hyena layers
        x = self.hyena_layers(x)
        
        # Biological context extraction (cell-level representation)
        context_vector = x.mean(dim=1)
        
        # Pretraining objectives
        if self.pretrain_head == 'masked_reconstruction':
            return self.head(x), self.context_proj(context_vector)
        else:
            return F.normalize(self.head(context_vector), self.context_proj(context_vector))





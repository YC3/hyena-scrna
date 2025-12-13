import math
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions import Bernoulli

from .layers import (
    HyenaBlock,
)

from einops import rearrange, repeat

from scgpt.model.model import (
    GeneEncoder,
    ContinuousValueEncoder,
    CategoryValueEncoder,
    BatchLabelEncoder,
    ExprDecoder,
    ClsDecoder,
    MVCDecoder,
    AdversarialDiscriminator,
    Similarity,
)
from scgpt.model.dsbn import DomainSpecificBatchNorm1d



class ScGeneHyena(nn.Module):

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        l_max: int, # max_length
        nlayers: int,
        d_hid: int,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = '<pad>',
        pad_value: int = 0,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: Union[bool, str] = False,
        input_emb_style: str = 'continuous',
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = 'cls',
        mvc_decoder_style: str = 'inner product',
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        hyena_order: int = 2,
        hyena_filter_order: int = 64,
        hyena_dropout: float = 0.1,
        hyena_filter_dropout: float = 0.1,
        hyena_num_heads: int = 1,
        conv_kernel_size: int = 3,
    ):

        """
        ScGeneHyena

        A Hyena-based scGPT-style model for single-cell gene expression with
        dual inputs: spliced and unspliced counts.

        Key differences from original scGPT (https://github.com/bowang-lab/scGPT):
          - Hyena backbone (HyenaBlock) instead of Transformer encoder.
          - Two value streams: spliced + unspliced, combined at embedding level.

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

        self.model_type = 'ScGeneHyena'
        self.d_model = d_model
        self.do_mvc = do_mvc
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
    
        if self.input_emb_style not in ['category', 'continuous', 'scaling']:
            raise ValueError(
                f"`input_emb_style` must be one of "
                f"['category', 'continuous', 'scaling'], got {self.input_emb_style}"
            )
        if self.domain_spec_batchnorm not in [True, 'dsbn', 'do_affine', 'batchnorm']:
            raise ValueError(
                f"`domain_spec_batchnorm` must be one of "
                f"[True, 'dsbn', 'do_affine', 'batchnorm'], got {self.domain_spec_batchnorm}"
            )
        if self.cell_emb_style not in ['cls', 'avg-pool', 'w-pool']:
            raise ValueError(f"Unknown `cell_emb_style`: {self.cell_emb_style}")

        if vocab is None or pad_token not in vocab:
            raise ValueError("A `vocab` containing `pad_token` is required.")
        padding_idx = vocab[pad_token]

        # gene encoder
        self.encoder = GeneEncoder(
            num_embeddings=ntoken,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

        # value encoder (spliced and unspliced counts)
        if self.input_emb_style == 'continuous':
            self.value_encoder_spliced = ContinuousValueEncoder(
                d_model=d_model,
                dropout=dropout,
            )
            self.value_encoder_unspliced = ContinuousValueEncoder(
                d_model=d_model,
                dropout=dropout,
            )
        elif self.input_emb_style == 'category':
            if n_input_bins is None or n_input_bins <= 0:
                raise ValueError(
                    "`n_input_bins` must be provided and > 0 for category style."
                )
            self.value_encoder_spliced = CategoryValueEncoder(
                n_bins=n_input_bins,
                d_model=d_model,
                padding_idx=pad_value,
            )
            self.value_encoder_unspliced = CategoryValueEncoder(
                n_bins=n_input_bins,
                d_model=d_model,
                padding_idx=pad_value,
            )
        else:
            self.value_encoder_spliced = nn.Identity()
            self.value_encoder_unspliced = nn.Identity()
        
        # batch encoder
        if self.use_batch_labels:
            if num_batch_labels is None:
                raise ValueError("`num_batch_labels` is required when `use_batch_labels` is True")
            self.batch_encoder = BatchLabelEncoder(
                num_batch_labels=num_batch_labels,
                d_model=d_model,
            )

        # domain-specific/global batch normalization
        if self.domain_spec_batchnorm in [True, 'dsbn', 'do_affine']:
            use_affine = self.domain_spec_batchnorm == 'do_affine'
            self.dsbn = DomainSpecificBatchNorm1d(
                num_features=d_model,
                num_domains=num_batch_labels,
                eps=6.1e-5,
                affine=use_affine,
            )

        elif self.domain_spec_batchnorm == 'batchnorm':
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        else:
            self.dsbn = None
            self.bn = None

        # the Hyena block
        self.hyena_block = HyenaBlock(
            d_model=d_model,
            l_max=l_max,
            order=hyena_order,
            filter_order=hyena_filter_order,
            dropout=hyena_dropout,
            filter_dropout=hyena_filter_dropout,
            num_heads=hyena_num_heads,
            depth=nlayers,
            conv_kernel_size=conv_kernel_size,
            hyena_dropout=hyena_dropout,
        )

        # decoders
        self.decoder = ExprDecoder(
            d_model=d_model,
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
        )
        
        self.cls_decoder = ClsDecoder(
            d_model=d_model,
            n_cls=n_cls,
            nlayers=nlayers_cls,
        )

        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model=d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )
        
        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model=d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )
        
        self.sim = Similarity(temp=0.5)
        self.creterion_cce = nn.CrossEntropyLoss()
        
        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)


    def _check_batch_labels(
        self,
        batch_labels: Optional[torch.Tensor],
    ) -> None:
        
        if self.use_batch_labels or self.domain_spec_batchnorm:
            if batch_labels is None:
                raise ValueError(
                    "`batch_labels` required when "
                    "`use_batch_labels` or `domain_spec_batchnorm` is True"
                )
        #else:
        if batch_labels is not None:
            raise ValueError(
                "`batch_labels` is provided, however this information won't be used "
                "since neither `use_batch_labels` or `domain_spec_batchnorm` is True"
            )


    def _get_cell_emb_from_layer(
        self,
        layer_output: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        layer_output: (batch, seq_len, d_model)
        weights: (batch, seq_len) used only for "w-pool".
        """

        if self.cell_emb_style == 'cls':
            cell_emb = layer_output[:, 0, :]
        elif self.cell_emb_style == 'avg-pool':
            cell_emb = torch.mean(layer_output, dim=1)
        else: 
            # w-pool
            if weights is None:
                raise ValueError("weights is required for w-pool")
            if weights.dim() != 2:
                raise ValueError("weights must be 2D (batch, seq_len)")
            cell_emb = torch.sum(
                layer_output*weights.unsqueeze(-1),
                dim=1,
            )
            cell_emb = F.normalize(cell_emb, p=2, dim=1)

        return cell_emb


    def _encode(
        self,
        src: torch.Tensor,
        values_spliced: torch.Tensor,
        values_unspliced: torch.Tensor,
        src_key_padding_mask: torch.BoolTensor,
        batch_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        Core encoder:
          gene tokens -> embeddings
          + spliced/unspliced value encodings
          -> (optional) domain/batch norm
          -> Hyena block
        """
        
        self._check_batch_labels(batch_labels)
        
        # Gene token embeddings: (batch, seq_len, d_model)
        src_emb = self.encoder(src)
        self.cur_gene_token_embs = src_emb
        
        # Value encodings
        v_s = self.value_encoder_spliced(values_spliced)
        v_u = self.value_encoder_unspliced(values_unspliced)
        
        if self.input_emb_style == "scaling":
            # treat v_s, v_u as scalar factors
            v_s = v_s.unsqueeze(-1)
            v_u = v_u.unsqueeze(-1)
            total_embs = src_emb * (v_s + v_u)
        else:
            # element-wise additive
            total_embs = src_emb + v_s + v_u
        
        # Domain-specific batch norm
        if hasattr(self, "dsbn"):
            # dsbn expects (batch, dim, seq), plus domain index
            # assume homogeneous batch domain (like scGPT implementation)
            domain_label = int(batch_labels[0].item())
            total_embs = self.dsbn(
                total_embs.permute(0, 2, 1),
                domain_label,
            ).permute(0, 2, 1)
        elif hasattr(self, "bn"):
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Hyena backbone
        x = self.hyena_block(
            x=total_embs,
            src_key_padding_mask=src_key_padding_mask,
        )
        
        return x  # (batch, seq_len, d_model)


    def forward(
        self,
        src: torch.Tensor,
        values_spliced: torch.Tensor,
        values_unspliced: torch.Tensor,
        src_key_padding_mask: torch.BoolTensor,
        batch_labels: Optional[torch.Tensor] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, torch.Tensor]:

        """
        Args:
            src: gene token ids, (batch, seq_len)
            values_spliced: spliced counts, (batch, seq_len)
            values_unspliced: unspliced counts, (batch, seq_len)
            src_key_padding_mask: bool mask, True for PAD, (batch, seq_len)
            batch_labels: optional batch labels, (batch,)
        """

        # encoding
        hyena_output = self._encode(
            src=src,
            values_spliced=values_spliced,
            values_unspliced=values_unspliced,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels,
        )
        
        output: Mapping[str, torch.Tensor] = {}
        
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, d_model)
            dec_input = torch.cat(
                [
                    hyena_output,
                    batch_emb.unsqueeze(1).repeat(1, hyena_output.shape[1], 1),
                ],
                dim=-1,
            )
        else:
            dec_input = hyena_output

        # decoding
        mlm_out = self.decoder(dec_input)

        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_out["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_out["pred"]
        else:
            output["mlm_output"] = mlm_out["pred"]

        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_out["zero_probs"]

        # extract cell embedding
        pool_weights = values_spliced if self.cell_emb_style == "w-pool" else None
        cell_emb = self._get_cell_emb_from_layer(hyena_output, pool_weights)
        output["cell_emb"] = cell_emb

        # CLS (cell type classification)
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)

        # CCE (contrastive cell embedding)
        if CCE:
            cell1 = cell_emb
            # second view via another pass
            hyena_output2 = self._encode(
                src=src,
                values_spliced=values_spliced,
                values_unspliced=values_unspliced,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels,
            )
            cell2 = self._get_cell_emb_from_layer(hyena_output2, pool_weights)
            
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1)
                    for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(cls1_list, cell1.contiguous())
                dist.all_gather(cls2_list, cell2.contiguous())
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2
                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)

        # MVC (masked value prediction for cell embeddings)
        if MVC and self.do_mvc:
            mvc_input = (
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=-1)
            )
            mvc_out = self.mvc_decoder(
                mvc_input,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_out["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_out["pred"]
            else:
                output["mvc_output"] = mvc_out["pred"]
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_out["zero_probs"]

        # ECS (elastic cell similarity)
        if ECS:
            cell_norm = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_norm, cell_norm.t())
            eye = torch.eye(cos_sim.size(0), device=cos_sim.device).bool()
            cos_sim = cos_sim.masked_fill(eye, 0.0)
            cos_sim = F.relu(cos_sim)
            output["loss_ecs"] = torch.mean(
                1 - (cos_sim - self.ecs_threshold)* 2
            )

        # DAB (adversarial discriminator)
        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)
        
        return output

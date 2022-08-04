import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from helper import *

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len = None,
        heads = 4,
        dim_head = 64,
        dropout = 0.0,
        gating = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, mask = None, attn_bias = None, context = None, context_mask = None, tie_dim = None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        context = default(context, x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        i, j = q.shape[-2], k.shape[-2]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # query / key similarities
        if exists(tie_dim):
            # as in the paper, for the extra MSAs
            # they average the queries along the rows of the MSAs
            # they named this particular module MSAColumnGlobalAttention

            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r = tie_dim), (q, k))
            q = q.mean(dim = 1)

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)
        if exists(attn_bias):
            dots = dots + attn_bias

        # masking
        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            try:
                mask = mask.to(torch.bool)
                dots = dots.masked_fill(~mask, mask_value)
            except:
                dots = dots.masked_fill(mask, mask_value)

        # attention
        dots = dots - dots.max(dim = -1, keepdims = True).values
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        # gating
        gates = self.gating(x)
        out = out * gates.sigmoid()
        # combine to out
        out = self.to_out(out)
        return out
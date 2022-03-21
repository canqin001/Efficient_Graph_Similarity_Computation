import torch
import torch.nn.functional as F
import torch.nn as nn

from math import ceil
from torch.nn import Linear, ReLU

import numpy as np

import pdb


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=1) # temperature=d_k ** 0.5

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q # 9 * 64

        q, attn = self.attention(q, k, v, mask=mask)
        q += residual

        return q, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(0, 1)) # attn 8 * 10, e.g.

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1) #self.dropout(F.softmax(attn, dim=-1)) # # # 8 * 10, the sum of every row is 1 after softmax, is <=1 after dropout
        output = torch.matmul(attn, v) # dim of output is same as q

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.batch_norm = nn.BatchNorm1d(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x

class CrossAttentionModule(torch.nn.Module):
    def __init__(self, args, dim_size):
        super(CrossAttentionModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()

    def setup_weights(self):
        self.multihead_attention = MultiHeadAttention(n_head=1, d_model=self.dim_size, d_k=self.dim_size, d_v=self.dim_size)
        self.mlp = PositionwiseFeedForward(d_in=self.dim_size, d_hid=self.dim_size)

    def forward(self, x_src, batch_src, x_tar, batch_tar, size=None):
        size = batch_src[-1] + 1 if batch_src[-1] == batch_tar[-1] else min(batch_src[-1], batch_tar[-1]) + 1
        score_batch = torch.zeros(size,1) # 128 * 1

        embed_batch_src = torch.zeros(x_src.size())
        embed_batch_tar = torch.zeros(x_tar.size())
        
        for i in range(size):
            loc_src = batch_src == i
            loc_tar = batch_tar == i
        
            feat_src_batch = x_src[loc_src,:]
            feat_tar_batch = x_tar[loc_tar,:]

            embed_src, _ = self.multihead_attention(feat_src_batch,feat_tar_batch,feat_tar_batch)
            embed_tar, _ = self.multihead_attention(feat_tar_batch,feat_src_batch,feat_src_batch)
            
            embed_batch_src[loc_src,:] = embed_src
            embed_batch_tar[loc_tar,:] = embed_tar

        embed_batch_tar = self.mlp(embed_batch_tar)
        embed_batch_src = self.mlp(embed_batch_src)
        return embed_batch_tar, embed_batch_src

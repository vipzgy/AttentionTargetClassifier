# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.init as init
from .BILSTM import BILSTM
from .Attention import Attention


class Vanilla(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_idx, label_size, embedding=None):
        super(Vanilla, self).__init__()
        self.config = config
        self.bilstm = BILSTM(config, embed_size, embed_dim, padding_idx, embedding)
        self.attention = Attention(config)
        self.linear_out = nn.Linear(config.hidden_size * 2, label_size)
        init.xavier_uniform(self.linear_out.weight)

    def forward(self, w, length, start, end):
        s_slice, targeted_slice, left_slice, right_slice, \
            s_mask, targeted_mask, left_mask, right_mask = self.bilstm(w, length, start, end)
        s = self.attention(s_slice, s_mask, targeted_slice)
        s = self.linear_out(s)
        return s

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
from .BILSTM import BILSTM
from .Attention import Attention


class Contextualized(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_idx, label_size, embedding=None):
        super(Contextualized, self).__init__()
        self.config = config
        self.bilstm = BILSTM(config, embed_size, embed_dim, padding_idx, embedding)
        self.attention_s = Attention(config)
        self.attention_left = Attention(config)
        self.attention_right = Attention(config)

        self.w = nn.Linear(config.hidden_size * 2 * 3, label_size)
        init.xavier_uniform(self.w.weight)

    def forward(self, w, length, start, end):
        s_slice, targeted_slice, left_slice, right_slice, \
            s_mask, targeted_mask, left_mask, right_mask = self.bilstm(w, length, start, end)
        s = self.attention_s(s_slice, s_mask, targeted_slice)
        sl = self.attention_left(left_slice, left_mask, targeted_slice)
        sr = self.attention_right(right_slice, right_mask, targeted_slice)

        ss = torch.cat([s, sl, sr], 1)

        result = self.w(ss)
        return result

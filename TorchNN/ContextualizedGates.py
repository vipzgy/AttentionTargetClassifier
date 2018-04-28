# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .BILSTM import BILSTM
from .Attention import Attention


class ContextualizedGates(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_idx, label_size, embedding=None):
        super(ContextualizedGates, self).__init__()
        self.config = config
        self.bilstm = BILSTM(config, embed_size, embed_dim, padding_idx, embedding)
        self.attention_s = Attention(config)
        self.attention_left = Attention(config)
        self.attention_right = Attention(config)

        self.w1 = nn.Linear(config.hidden_size * 4, config.hidden_size * 2)
        init.xavier_uniform(self.w1.weight)
        self.w2 = nn.Linear(config.hidden_size * 4, config.hidden_size * 2)
        init.xavier_uniform(self.w2.weight)
        self.w3 = nn.Linear(config.hidden_size * 4, config.hidden_size * 2)
        init.xavier_uniform(self.w3.weight)

        self.w4 = nn.Linear(config.hidden_size * 2, label_size)
        init.xavier_uniform(self.w4.weight)

    def forward(self, w, length, start, end):
        s_slice, targeted_slice, left_slice, right_slice, \
            s_mask, targeted_mask, left_mask, right_mask = self.bilstm(w, length, start, end)
        s = self.attention_s(s_slice, s_mask, targeted_slice)
        sl = self.attention_left(left_slice, left_mask, targeted_slice)
        sr = self.attention_right(right_slice, right_mask, targeted_slice)

        ht = torch.mean(targeted_slice, 1)
        z = self.w1(torch.cat([s, ht], 1))
        zl = self.w2(torch.cat([sl, ht], 1))
        zr = self.w3(torch.cat([sr, ht], 1))

        zz = torch.cat([torch.unsqueeze(z, 0), torch.unsqueeze(zl, 0), torch.unsqueeze(zr, 0)], 0)
        zz = F.softmax(zz, 0)
        z = torch.squeeze(zz[0], 0)
        zl = torch.squeeze(zz[1], 0)
        zr = torch.squeeze(zz[2], 0)

        ss = torch.mul(z, s) + torch.mul(zl, sl) + torch.mul(zr, sr)
        result = self.w4(ss)
        return result

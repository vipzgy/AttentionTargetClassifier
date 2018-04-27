# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Vanilla(nn.Module):
    def __init__(self, config, ):
        super(Vanilla, self).__init__()

        self.w1 = nn.Linear()
        self.u = nn.Parameter()
        self.w2 = nn.Linear()

    def forward(self, h, h_length, ht):
        return
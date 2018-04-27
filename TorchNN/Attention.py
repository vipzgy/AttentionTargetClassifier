# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.w1 = nn.Linear(config.hidden_size * 4, config.attention_size, bias=True)
        self.u = nn.Linear(config.attention_size, 1, bias=False)

    def forward(self, h, h_mask, ht):
        ht = torch.mean(ht, 1)
        ht = torch.unsqueeze(ht, 1)

        ht = ht.repeat(1, h.size(1), 1)
        h_cat = torch.cat([h, ht], 2)
        h_mask = Variable(torch.FloatTensor(h_mask))
        h_mask1 = h_mask.unsqueeze(2).repeat(1, 1, self.config.hidden_size * 4)
        h_cat = torch.mul(h_cat, h_mask1)

        h_cat = self.w1(h_cat)
        h_cat = F.tanh(h_cat)
        h_mask2 = h_mask.unsqueeze(2).repeat(1, 1, self.config.attention_size)
        h_cat = torch.mul(h_cat, h_mask2)
        beta = self.u(h_cat)

        beta = torch.squeeze(beta, 2)
        beta_0 = Variable(torch.zeros(beta.size(0), beta.size(1)))
        beta_0 += -1e20
        h_mask = h_mask * -1 + 1
        beta = beta.masked_scatter(h_mask.type(torch.ByteTensor), beta_0.masked_select(h_mask.type(torch.ByteTensor)))
        alpha = F.softmax(beta, dim=1)

        alpha = torch.unsqueeze(alpha, 2)
        alpha = alpha.repeat(1, 1, self.config.hidden_size * 2)
        s = torch.mul(h, alpha)
        s = torch.sum(s, 1)
        return s
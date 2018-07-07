# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.w1 = nn.Linear(config.hidden_size * 4, config.attention_size, bias=True)
        init.xavier_uniform(self.w1.weight)
        self.u = nn.Linear(config.attention_size, 1, bias=False)
        init.xavier_uniform(self.u.weight)

    def forward(self, h, h_mask, ht):
        ht = torch.mean(ht, 1)

        ht = torch.unsqueeze(ht, 1)
        ht = ht.repeat(1, h.size(1), 1)
        h_cat = torch.cat([h, ht], 2)
        if self.config.use_cuda:
            # function 1
            # h_mask = Variable(torch.FloatTensor(h_mask)).cuda()
            # function 2
            h_mask = Variable(torch.IntTensor(h_mask)).cuda()
            zeros_hidden = Variable(torch.zeros(h_cat.size(0), h_cat.size(1), self.config.hidden_size * 4)).cuda()
            zeros_attention = Variable(torch.zeros(h_cat.size(0), h_cat.size(1), self.config.attention_size)).cuda()
            h_mask = torch.abs(h_mask - 1).type(torch.cuda.ByteTensor)
        else:
            # function 1
            # h_mask = Variable(torch.FloatTensor(h_mask))
            # function 2
            h_mask = Variable(torch.IntTensor(h_mask))
            zeros_hidden = Variable(torch.zeros(h_cat.size(0), h_cat.size(1), self.config.hidden_size * 4))
            zeros_attention = Variable(torch.zeros(h_cat.size(0), h_cat.size(1), self.config.attention_size))
            h_mask = torch.abs(h_mask - 1).type(torch.ByteTensor)
        # function 1
        # h_mask1 = h_mask.unsqueeze(2).repeat(1, 1, self.config.hidden_size * 4)
        # h_cat = torch.mul(h_cat, h_mask1)
        # function 2
        # h_mask = torch.abs(h_mask - 1).type(torch.ByteTensor)
        h_mask = torch.unsqueeze(h_mask, 2)
        h_cat = h_cat.masked_scatter(h_mask, zeros_hidden.masked_select(h_mask))

        h_cat = self.w1(h_cat)
        h_cat = F.tanh(h_cat)
        h_cat = h_cat.masked_scatter(h_mask, zeros_attention.masked_select(h_mask))
        beta = self.u(h_cat)

        beta = torch.squeeze(beta, 2)
        if self.config.use_cuda:
            beta_0 = Variable(torch.zeros(beta.size(0), beta.size(1))).cuda()
        else:
            beta_0 = Variable(torch.zeros(beta.size(0), beta.size(1)))
        beta_0 += -1e20
        h_mask = torch.squeeze(h_mask, 2)
        if self.config.use_cuda:
            beta = beta.masked_scatter(h_mask, beta_0.masked_select(h_mask))
        else:
            beta = beta.masked_scatter(h_mask, beta_0.masked_select(h_mask))
        alpha = F.softmax(beta, dim=1)

        # 这里就可以作图了，看看attention的值怎么样

        alpha = torch.unsqueeze(alpha, 2)
        alpha = alpha.repeat(1, 1, self.config.hidden_size * 2)
        s = torch.mul(h, alpha)
        s = torch.sum(s, 1)
        return s

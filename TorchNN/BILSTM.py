# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BILSTM(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_idx, embedding=None):
        super(BILSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(embed_size, embed_dim, padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.dropout = nn.Dropout(config.dropout_embed)
        self.bilstm = nn.LSTM(embed_dim, config.hidden_size, num_layers=config.num_layers,
                              dropout=config.dropout_rnn, bidirectional=True)

    def forward(self, w, length, start, end):
        e = self.embedding(w)
        e = self.dropout(e)

        h = pack_padded_sequence(e, length)
        h, _ = self.bilstm(h)
        h, _ = pad_packed_sequence(h)

        # ht, [h0,...,hn] except ht, [h0,...ht-1], [ht+1,...,hn]
        h_tem = h.transpose(0, 1)
        x_size = h_tem.size(0)
        y_size = h_tem.size(1)
        z_size = h_tem.size(2)
        h_tem = h_tem.contiguous().view(x_size * y_size, z_size)
        h_tem = torch.cat([h_tem, Variable(torch.zeros(1, z_size))], 0)

        # list[list] 与h相对应的坐标
        padding_id = x_size * y_size
        left_max = max(start)
        left = [[idx * y_size + j if j < x else padding_id for j in range(left_max)] for idx, x in enumerate(start)]

        right_max = max([l - e - 1 for l, e in zip(length, end)])
        right = [[idx * y_size + j + e + 1 if j < l - e - 1 else padding_id for j in range(right_max)] for
                 idx, (e, l) in enumerate(zip(end, length))]

        targeted_length = [e - s + 1 for s, e in zip(start, end)]
        targeted_max = max(targeted_length)
        targeted = [[idx * y_size + j + s if j < e - s + 1 else padding_id for j in range(targeted_max)] for
                    idx, (s, e) in enumerate(zip(start, end))]

        s_max = max([l - e + s - 1 for l, s, e in zip(length, start, end)])
        s = []
        s_mask = []
        for idx in range(x_size):
            s_t = []
            s_t_mask = []
            for idy in range(y_size):
                if idy < length[idx]:
                    if idy < start[idx] or idy > end[idx]:
                        s_t.append(idx * y_size + idy)
                        s_t_mask.append(1)
            for i in range(s_max - len(s_t)):
                s_t.append(padding_id)
                s_t_mask.append(0)
            s.append(s_t)
            s_mask.append(s_t_mask)

        # list 与h_tem想对应的坐标
        left_index = []
        right_index = []
        targeted_index = []
        s_index = []
        for l, r, tar, ss in zip(left, right, targeted, s):
            left_index += l
            right_index += r
            targeted_index += tar
            s_index += ss

        # mask
        left_mask = [[1 if j < x else 0 for j in range(left_max)] for idx, x in enumerate(start)]
        right_mask = [[1 if j < l - e - 1 else 0 for j in range(right_max)] for
                 idx, (e, l) in enumerate(zip(end, length))]
        targeted_mask = [[1 if j < e - s + 1 else 0 for j in range(targeted_max)] for
                    idx, (s, e) in enumerate(zip(start, end))]

        # slice
        left_slice = torch.index_select(h_tem, 0, Variable(torch.LongTensor(left_index)))
        left_slice = left_slice.view(x_size, left_max, z_size)
        right_slice = torch.index_select(h_tem, 0, Variable(torch.LongTensor(right_index)))
        right_slice = right_slice.view(x_size, right_max, z_size)
        targeted_slice = torch.index_select(h_tem, 0, Variable(torch.LongTensor(targeted_index)))
        targeted_slice = targeted_slice.view(x_size, targeted_max, z_size)
        s_slice = torch.index_select(h_tem, 0, Variable(torch.LongTensor(s_index)))
        s_slice = s_slice.view(x_size, s_max, z_size)

        return s_slice, targeted_slice, left_slice, right_slice, s_mask, targeted_mask, left_mask, right_mask

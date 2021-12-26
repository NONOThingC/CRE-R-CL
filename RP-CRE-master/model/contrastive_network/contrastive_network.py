# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base_model import base_model
from transformers import BertModel, BertConfig
LARGE_NUM = 1e9
class ContrastiveNetwork(base_model):

    def __init__(self, config,encoder,hidden_size):

        super(ContrastiveNetwork, self).__init__()
        self.encoder=encoder
        self.hidden_size = hidden_size
        self.dim_trans = nn.Linear(config.encoder_output_size, hidden_size)
        self.output_size = config.encoder_output_size  # for classification number
        self.projector = nn.Linear(hidden_size, self.output_size)
        self.emb_trans = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.projector1 = nn.Linear(self.output_size, self.output_size // 2)
        self.temperature = config.temperature

    def forward(self, enc_inp, proj_inp, comparison):
        mid_hidden = self.encoder(enc_inp)
        if mid_hidden.shape[1] != self.hidden_size:
            mid_hidden = self.dim_trans(mid_hidden)
        proj_inp = self.emb_trans(proj_inp)
        hidden = self.projector1(self.relu(self.projector(torch.cat([mid_hidden, proj_inp], dim=0))))

        hidden = F.normalize(hidden, dim=-1, p=2)
        # hidden = torch.linalg.norm(hidden, dim=-1)
        hidden1, hidden2 = torch.split(hidden, [len(enc_inp), len(proj_inp)], dim=0)
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden2, -1, -2)) / self.temperature  # B1*B2
        logits_aa = logits_aa + (comparison == 0).float() * (-LARGE_NUM)  # mask#B1 * B2
        return logits_aa

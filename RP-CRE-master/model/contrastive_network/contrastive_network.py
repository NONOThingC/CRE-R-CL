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
        self.projector=nn.Linear(hidden_size,self.output_size)
        self.relu = nn.ReLU()
        self.projector1 = nn.Linear(self.output_size, self.output_size // 2)
        self.temperature = config.temperature

    def forward(self, enc_inp, proj_inp, comparison):
        mid_hidden = self.encoder(enc_inp)
        if mid_hidden.shape[0] != self.hidden_size:
            mid_hidden = self.dim_trans(mid_hidden)
        hidden = self.projector1(self.relu(self.projector(torch.concat([mid_hidden, proj_inp], dim=0))))
        hidden = torch.linalg.norm(hidden, dim=-1)
        hidden1, hidden2 = torch.split(hidden, 2, dim=0)
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden2, -1, -2)) / self.temperature
        logits_aa = logits_aa + (~comparison) * (-LARGE_NUM)  # mask
        return logits_aa

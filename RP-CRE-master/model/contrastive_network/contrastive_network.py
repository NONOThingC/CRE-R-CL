# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base_model import base_model
from transformers import BertModel, BertConfig

class ContrastiveNetwork(base_model):

    def __init__(self, config,encoder,hidden_size):
        super(ContrastiveNetwork, self).__init__()
        self.encoder=encoder
        self.hidden_size=hidden_size
        enc_out=config#替换成encoder output
        self.dim_trans=nn.Linear(enc_out,hidden_size)
        self.output_size = config.encoder_output_size# for classification number
        self.projector=nn.Linear(hidden_size,self.output_size)
        self.relu=nn.ReLU()

    def forward(self,enc_inp,proj_inp):
        mid_hidden=self.encoder(enc_inp)
        if mid_hidden.shape[0]!=self.hidden_size:
            mid_hidden=self.dim_trans(mid_hidden)
        output=self.relu(self.projector(torch.concat([mid_hidden,proj_inp],dim=0)))

        return output
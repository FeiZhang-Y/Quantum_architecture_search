import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class TransformerPredictor(nn.Module):
    def __init__(self, arg):
        super(TransformerPredictor, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=arg.hidden_dim, nhead=arg.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=arg.num_layers)
        self.embedding_matrix = nn.Parameter(torch.empty(10, arg.hidden_dim))
        nn.init.uniform_(self.embedding_matrix)
        self.positional_mlp = nn.Linear(41, arg.hidden_dim)

        self.fc1 = nn.Linear(arg.hidden_dim, 1)
        if arg.dropout:
            self.drop = nn.Dropout(p = arg.dropratio)
        else:
            self.drop = nn.Dropout(p = 0)
        self.fc2 = nn.Linear(41, 1)

    def forward(self, ops, adj):
        # encoding = self.positional_encoding(ops)
        operational = torch.matmul(ops, self.embedding_matrix)
        positional = self.positional_mlp(adj)
        positional = F.leaky_relu(positional,negative_slope=0.01, inplace=True)
        # positional = F.softmax(positional)
        input_feature = operational + positional
        output_feature = self.transformer_encoder(input_feature)
        out = self.fc1(output_feature)
        out = self.drop(out)
        out = out.view(len(out), -1)
        out = F.leaky_relu(out,negative_slope=0.01, inplace=True)
        # out = F.softmax(out)
        out = self.fc2(out)
        out = out.view(len(out))
        return out

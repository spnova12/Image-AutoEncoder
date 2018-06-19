import torch
import torch.nn as nn
import torch.nn.functional as F
from _models import *
"""
one 
"""

class Auto_encoder(nn.Module):
    def __init__(self):
        super(Auto_encoder, self).__init__()
        D_in = 48*48
        H = 20*20

        self.encoder = nn.Linear(D_in, H)
        self.decoder = nn.Linear(H, D_in)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, noise=None):
        out = x.view(-1, self.num_flat_features(x))
        out = self.encoder(out)
        out = self.decoder(out)
        out = F.tanh(out)
        out = out.view(out.size()[0], 1, 48, 48)
        return out



from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import grad
from .functions import get_activation

import logging

logger = logging.getLogger(__name__)

class RatioModel(nn.Module):

    def __init__(self, n_observables, n_hidden, activation="relu", dropout_prob=0.5):

        super(RatioModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation(activation)
        self.dropout_prob = dropout_prob

        # Build network
        #self.layers = nn.ModuleList()
        #n_last = n_observables

        self.ll1 = nn.Linear(n_observables, 10)  # 5x5 image dimension
        self.ll2 = nn.Linear(10, 10)
        self.ll3 = nn.Linear(10, 1)



        # Hidden layers
#        for n_hidden_units in n_hidden:
#            if self.dropout_prob > 1.0e-9:
#                self.layers.append(nn.Dropout(self.dropout_prob))
#            self.layers.append(nn.Linear(n_last, n_hidden_units))
#            n_last = n_hidden_units

        # Log r layer
#        if self.dropout_prob > 1.0e-9:
#            self.layers.append(nn.Dropout(self.dropout_prob))
#        self.layers.append(nn.Linear(n_last, 1))


    def forward(self, x: torch.Tensor):
        s_hat = x
        s_hat = F.relu(self.ll1(s_hat))
        s_hat = F.relu(self.ll2(s_hat))
        s_hat = self.ll3(s_hat)

#        for i, layer in enumerate(self.layers):
#            if i > 0:
#                s_hat = self.activation(s_hat)
#            s_hat = layer(s_hat) 
        r_hat = (1 - torch.sigmoid(s_hat)) / torch.sigmoid(s_hat)
        return r_hat, s_hat

    def to(self, *args, **kwargs):
        self = super(RatioModel, self).to(*args, **kwargs)

        #for i, layer in enumerate(self.layers):
        self.ll1 = self.ll1.to(*args, **kwargs)
        self.ll2 = self.ll2.to(*args, **kwargs)
        self.ll3 = self.ll3.to(*args, **kwargs)

        return self


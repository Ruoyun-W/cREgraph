import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv,summary
import random
import sys
import copy
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from tqdm import tqdm
import json
from itertools import combinations
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.patches as mpatches



class gnn_network(torch.nn.Module):

    def __init__(self, index_size, gcn_input_size, gcn_hidden_size, gcn_output_size, batch_size, sample_size):

        super(gnn_network, self).__init__()

        self.conv1 = GCNConv(gcn_input_size, gcn_hidden_size)
        self.conv2 = GCNConv(gcn_hidden_size, gcn_output_size)

        self.matrix = Variable(torch.randn(gcn_output_size, gcn_output_size).type(torch.FloatTensor), requires_grad=True)
        self.linear = torch.nn.Linear(1,2)

        self.bilinear = nn.Bilinear(gcn_output_size, gcn_output_size, 1)

        self.transform1 = nn.Linear(index_size, gcn_input_size)
        self.transform2 = nn.Linear(768, gcn_input_size)

        self.gcn_output_size = gcn_output_size

        self.sample_size = sample_size
        self.index_size = index_size      

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        attributes1 = x[:, :self.index_size].float()
        attributes2 = x[:, self.index_size:].float()

        x = self.transform1(attributes1) + self.transform2(attributes2)
    
        x = self.conv1(x.float(), edge_index)
  
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)


        ypred = []
        for i in range(x.size(0) // self.sample_size):
            embedings = x[i * self.sample_size:(i + 1) * self.sample_size, :]
            ypred.append(self.linear(torch.matmul(torch.matmul(embedings, self.matrix), embedings.t()).unsqueeze(-1)).view(-1, 2))
            # ypred.append(self.linear(self.bilinear(emb1, emb2)).unsqueeze(-1).view(-1, 2))
        return torch.stack(ypred).view(-1,2)
    

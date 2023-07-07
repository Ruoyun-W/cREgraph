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


def make_file_adj_graph(edge_mat, graph_title, path_out):
  print("Plot section...")
  values_mat = np.unique(edge_mat.ravel())
  color_dict = {-1: 'lightblue',
              0: 'darkslateblue',
              1: 'yellow'}
  colors=[color_dict[val] for val in list(values_mat)]
  im = plt.imshow(edge_mat,interpolation="none",cmap=ListedColormap(colors))
  plt.title(graph_title)
  colors = [ im.cmap(im.norm(value)) for value in values_mat]
  patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values_mat[i]) ) for i in range(len(values_mat)) ]
  plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
  plt.show()
  # plt.savefig(path_out) 

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
    

def train_gnn_model(dataset_train, dataset_test, config_path):
  

  config_file = open(config_path)
  config_json = json.load(config_file)


  feature = torch.tensor(config_json["feature_inp"])
  model = gnn_network(feature.size(0), config_json["gcn_input_size"], config_json["gcn_hidden_size"], config_json["gcn_output_size"],
                        config_json["batch_size"], config_json["sample_size"])
  criterion = torch.nn.CrossEntropyLoss(ignore_index =-1)
  optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)


  losses_train = []
  losses_test = []
  for epoch in range(config_json["num_epochs"]):
    count = 0
    loss = 0
    flag = True
    for data in dataset_train:
        pred_y = model(data)
        loss = loss + criterion(pred_y, data.y)
        count = count + 1
     

        if(count == config_json["backward_lim"]):
            loss.backward()
            optimizer.step()
            if(flag):
              losses_train.append(loss.item()) 
            flag = False
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            optimizer.zero_grad()
            loss = 0
            count = 0
    loss_test = 0
    for data in dataset_test:
      pred_y = model(data)
      loss_test = loss_test + criterion(pred_y, data.y)
    losses_test.append(loss_test.item())

    if(epoch==0):
        print(summary(model,data)) 
    if(loss != 0):
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        optimizer.zero_grad()
  return model, losses_train, losses_test


def test_model(model_cRE_edge_predictior, dataset_test, config_path):

  
  config_file = open(config_path)
  config_json = json.load(config_file)

  y_true = []
  y_pred = []

  cnt = 0

  with torch.no_grad():    
    for data in dataset_test:
      pred_y = model_cRE_edge_predictior(data)
      v, i = torch.max(pred_y, dim = 1)
      y_true += data.y.tolist()
      y_pred += i
      cnt += 1
      print((data.y == -1).sum())
      if(cnt <= 2):
        mat_y = np.reshape(data.y[0:10000], (100, 100))
        mat_pred_y = np.reshape(y_pred[0:10000], (100, 100))
        mat_y = mat_y.numpy()
        make_file_adj_graph(mat_y, "original_sample_100", "../../Result/sample100_original.png")
        make_file_adj_graph(mat_y, "pred_sample_100", "../../Result/sample100_pred.png")

        

  y_pred_samples = [y_pred[i] for i in range(len(y_pred)) if y_true[i] != -1] 
  y_true_samples = [var for var in y_true if var != -1] 
  f1 = f1_score(y_true_samples, y_pred_samples, average='macro')
  print("f1 score is: ", f1)      

  return y_true, y_pred
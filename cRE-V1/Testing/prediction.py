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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def make_file_adj_graph(edge_mat, graph_title, path_out):
  values_mat = np.unique(edge_mat.ravel())
  color_dict = {-1: 'lightblue',
              0: 'darkslateblue',
              1: 'yellow'}
  colors=[color_dict[val] for val in list(values_mat)]
  im = plt.imshow(edge_mat,interpolation="none",cmap=ListedColormap(colors))
  plt.title(graph_title)
  colors = [ im.cmap(im.norm(value)) for value in values_mat]
  patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values_mat[i]) ) for i in range(len(values_mat)) ]
  plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
  plt.show()
  plt.savefig(path_out) 

def predict_edges(cre_dataset, model_cRE_edge_predictor, index_subgraph1, index_subgraph2, all_attributes, num_nodes, 
                  config_json, cRE_edge_matrix_input):
  sample_size = 2 * num_nodes
  edge_list_node1 = []
  edge_list_node2 = []

  output_predicted = [0] * (sample_size ** 2)


  node_set1 = list(range(index_subgraph1 * num_nodes, (index_subgraph1 + 1) * num_nodes))
  node_set2 = list(range(index_subgraph2 * num_nodes, (index_subgraph2 + 1) * num_nodes))
  node_set = node_set1 + node_set2 


  attribute_dataset = []

  node_part1 = range(0, num_nodes)
  node_part2 = range(num_nodes, num_nodes * 2)  

  for node1 in node_part1:
    for node2 in node_part2: 
      if(node_set[node1] >= cre_dataset.cRE_number_of_nodes or  node_set[node2] >= cre_dataset.cRE_number_of_nodes):
        continue
      if(cRE_edge_matrix_input[node_set[node1]][node_set[node2]] == 1):
        edge_list_node1.extend([node1, node2])
        edge_list_node2.extend([node2, node1])

      output_predicted[node1 * sample_size + node2] = cre_dataset.cRE_edge_matrix[node_set[node1]][node_set[node2]]
      output_predicted[node2 * sample_size + node1] = cre_dataset.cRE_edge_matrix[node_set[node2]][node_set[node1]]
  
  for indx_node in range(sample_size):
    if(node_set[node1] < cre_dataset.cRE_number_of_nodes):
      attribute_dataset.append(all_attributes[node_set[indx_node]].tolist())
    else:
      attribute_dataset.append(all_attributes[0].tolist())
  
  feature = torch.tensor(config_json.feature_inp)
  all_features_index = torch.cat([feature, torch.Tensor(list(range(11,779))).long()],dim=0)
  attribute_dataset = torch.tensor(np.array(attribute_dataset))
  attribute_dataset = torch.index_select(attribute_dataset, 1, all_features_index)
  edge_index = torch.tensor([edge_list_node1, edge_list_node2], dtype=torch.long)

  data_need_pred = Data(x = torch.tensor(np.array(attribute_dataset)), edge_index = torch.tensor(np.array(edge_index)).long(),
                         y = torch.tensor(np.array(output_predicted)).long())

  pred_y = model_cRE_edge_predictor(data_need_pred)
  softmax_pred = F.softmax(pred_y)
  v, ind = torch.max(softmax_pred, dim = 1)
  softmax_pred = softmax_pred.tolist()



  diff_prob = [tupple_prob[1] - tupple_prob[0] for tupple_prob in softmax_pred]

  is_edge = [1 if softmax_pred[k][1] - softmax_pred[k][0] >= 0.5 else 0 for k in range(len(ind))]

  pred_edge_index = [var[0] - 1 for var in enumerate(is_edge, 1) if var[1] == 1]

  matrix_edges_predicted_subgraph = torch.zeros(sample_size, sample_size)

  for ind in pred_edge_index:
    node1 = ind % sample_size 
    node2 = ind // sample_size
    ind2 = node2 * sample_size + node1
    if(not ind2 in pred_edge_index or node1 == node2):
      continue 

    matrix_edges_predicted_subgraph[node1, node2] = 1

  matrix_out = torch.zeros(num_nodes, num_nodes)

  for i in range(0, 50):
    for j in range(50, 100):
      matrix_out[i][j - 50] = matrix_edges_predicted_subgraph[i][j]



  return matrix_out

def predict_whole_graph(cre_dataset, model_cRE_edge_predictor, num_nodes_each_part_pred, path_out, path_config, pic_path, cRE_edge_matrix_input):

  config_file = open(path_config)
  config_json = json.load(path_config)
  prediction = torch.zeros(10000, 10000)
  number_nodes_each_part = config_json.num_nodes_each_part
  sample_size = 2 * config_json.num_nodes_each_part

  x_attributes = copy.deepcopy(cre_dataset.cRE_attributes)
  x_attributes = torch.cat([x_attributes, torch.tensor(cre_dataset.cRE_DNA)], 1)
  x_attributes = torch.cat([x_attributes, x_attributes[:10000 - x_attributes.size(0)]])

  for i in tqdm(range(0, 10000 // 50)):
    for j in range(0, i + 1):
      predicted_subgraph = predict_edges(cre_dataset, model_cRE_edge_predictor, i, j, x_attributes, c
                                         onfig_json.num_nodes_each_part, config_json, cRE_edge_matrix_input)
      prediction[i * number_nodes_each_part:(i+1) * number_nodes_each_part, 
                 j * number_nodes_each_part : (j+1) * number_nodes_each_part] = predicted_subgraph
      prediction[j * number_nodes_each_part:(j+1) * number_nodes_each_part, 
                 i * number_nodes_each_part : (i+1) * number_nodes_each_part] = predicted_subgraph.t()
      if(i % 100 == 0 and j % 100 == 0):
        make_file_adj_graph(cre_dataset.cRE_edge_matrix, "cRE egdes before prediction")
        make_file_adj_graph(prediction, "cRE edges after prediciton")


  make_file_adj_graph(cre_dataset.cRE_edge_matrix, "cRE egdes before prediction", pic_path)
  make_file_adj_graph(prediction, "cRE edges after prediciton", pic_path)

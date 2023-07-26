import numpy as np 
import networkx as nx
import pandas as pd
import torch 
import tqdm 
import random
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import json 

FEATURE_LIST = [3,5,6,7,8,9,10,11,12,13,14]


class InitialData:

    def read_cre_attributes(self, path_attributes):

        cre_attributes_read = pd.read_table(path_attributes, header = None , sep=" ")
        cre_attributes_selected = cre_attributes_read.iloc[: , FEATURE_LIST]
        cre_attributes_selected = cre_attributes_selected.to_numpy()
        cre_attributes = torch.tensor(cre_attributes_selected, dtype = torch.float)
        return cre_attributes,  cre_attributes.size()[0]

    def read_cre_DNA_attributes(self, path_dna):

        cre_dna = np.load(path_dna)
        return cre_dna

    def read_cre_is_edge(self, path_edges, cre_number_of_nodes):

        cre_edge = pd.read_table(path_edges, header = None, sep=" ")
        cre_number_of_edges = len(cre_edge)
        cre_edge = torch.tensor([cre_edge[:][0], cre_edge[:][1]], dtype = torch.long)

        edge_pairs = cre_edge
        cre_edge_matrix = 2 * torch.eye(cre_number_of_nodes, cre_number_of_nodes) - 1
        for i in range(int(len(cre_edge[1]))):
            cre_edge_matrix[cre_edge[0][i]][cre_edge[1][i]] = 1
            cre_edge_matrix[cre_edge[1][i]][cre_edge[0][i]] = 1
        return edge_pairs, cre_edge_matrix, cre_number_of_edges

    def load_cre_graph(self, path_cre_attributes, path_dna, path_edges, config_path):

        cre_attributes, cre_number_of_nodes = self.read_cre_attributes(path_cre_attributes)
        cre_dna = self.read_cre_DNA_attributes(path_dna)
        cre_edge_pairs, cre_edge_matrix, cre_number_of_edges = self.read_cre_is_edge(path_edges, cre_number_of_nodes)
        config_file = open(config_path)
        config_json = json.load(config_file)

        return cre_attributes, cre_dna, cre_edge_pairs, cre_edge_matrix

    def load_graph_data(self, path_directory, dataset_name, config_path):

        if(dataset_name == "cre"):
            path_cre_attributes = path_directory + "/LZ_cRE_attributes/LZ_chr1.txt"
            path_dna = path_directory + "/cRE_DNA/chr1.npy"
            path_edge = path_directory + "/LZ_cREedge/LZ_t4_chr1.txt"
            return self.load_cre_graph(path_cre_attributes, path_dna, path_edge, config_path)
    
class NewDataSet:

    def read_cre_attributes(self, path_attributes):
        cre_attributes_read = pd.read_table(path_attributes, header = None , sep=" ")
        print(cre_attributes_read)

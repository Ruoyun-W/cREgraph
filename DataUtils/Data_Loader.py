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


def read_cre_attributes(path_attributes):

    cre_attributes_read = pd.read_table(path_attributes, header = None , sep=" ")
    cre_attributes_selected = cre_attributes_read.iloc[: , FEATURE_LIST] ## why these columns? what about some feature selection algorithms?
    cre_attributes_selected = cre_attributes_selected.to_numpy()
    # print("HELLOOOO")
    # cre_attributes = torch.tensor(StandardScaler().fit_transform(cre_attributes_selected.to_numpy()), dtype=torch.float)
    cre_attributes = torch.tensor(cre_attributes_selected, dtype = torch.float)
    return cre_attributes,  cre_attributes.size()[0]

def read_cre_DNA_attributes(path_dna):

    cre_dna = np.load(path_dna)
    return cre_dna

def read_cre_is_edge(path_edges, cre_number_of_nodes):

    cre_edge = pd.read_table(path_edges, header = None, sep=" ")
    cre_number_of_edges = len(cre_edge)
    cre_edge = torch.tensor([cre_edge[:][0], cre_edge[:][1]], dtype = torch.long)

    edge_pairs = cre_edge
    cre_edge_matrix = 2 * torch.eye(cre_number_of_nodes, cre_number_of_nodes) - 1

    for i in range(int(len(cre_edge[1]))):
        cre_edge_matrix[cre_edge[0][i]][cre_edge[1][i]] = 1
        cre_edge_matrix[cre_edge[1][i]][cre_edge[0][i]] = 1
    return edge_pairs, cre_edge_matrix, cre_number_of_edges

def load_cre_graph(path_cre_attributes, path_dna, path_edges):

    cre_attributes, cre_number_of_nodes = read_cre_attributes(path_cre_attributes)
    cre_dna = read_cre_DNA_attributes(path_dna)
    cre_edge_pairs, cre_edge_matrix, cre_number_of_edges = read_cre_is_edge(path_edges, cre_number_of_nodes)
    config_file = open('../DataUtils/config.json')
    config_json = json.load(config_file)

    return cre_attributes, cre_dna, cre_edge_pairs, cre_edge_matrix

    # if(config_json["is_negative"]):
    #     cre_edge_matrix_input = add_negative_samples(config_json["portion_hidden_edges"], config_json["ratio_negative"], cre_number_of_edges, 
    #                                                  cre_number_of_nodes, cre_edge_matrix)
    # else:
    #     cre_edge_matrix_input = cre_edge_matrix.copy()
    
    # generate_random_subgraph(cre_edge_matrix_input, config_json["radius"], config_json["random_size"],
    #                         neighbor_inp, cre_number_of_nodes, cre_number_of_edges, 
    #                         cre_edge_matrix_input, cre_attributes, cre_DNA)


def load_graph_data(path_directory, dataset_name):

    if(dataset_name == "cre"):
        path_cre_attributes = path_directory + "/LZ_cRE_attributes/LZ_chr1.txt"
        path_dna = path_directory + "/cRE_DNA/chr1.npy"
        path_edge = path_directory + "/LZ_cREedge/LZ_t4_chr1.txt"
        return load_cre_graph(path_cre_attributes, path_dna, path_edge)

# def add_negative_samples(portion_hidden_edges, portion_negative_to_positive, cre_number_of_edges, 
#                          cre_number_of_nodes, cre_edge_matrix): 
  
#     portion = portion_hidden_edges
#     cre_edge_matrix_input =  torch.clone(cre_edge_matrix)
#     number_edge_samples = cre_number_of_edges * portion_hidden_edges

#     number_negative_samples = cre_number_of_edges * portion_negative_to_positive
#     count_negative = 0

#     have_already_sampled = torch.eye(cre_number_of_nodes, cre_number_of_nodes)

#     num_node_zero_base = cre_number_of_nodes - 1

#     while(count_negative < number_negative_samples):
        
#         edge_index_in_matrix = random.randint(0, num_node_zero_base * num_node_zero_base)
#         node1 = edge_index_in_matrix // num_node_zero_base
#         node2 = edge_index_in_matrix % num_node_zero_base
#         if(cre_edge_matrix_input[node1][node2] == -1 and have_already_sampled[node1][node2] == 0): 

#             cre_edge_matrix_input[node1][node2] = 0
#             cre_edge_matrix_input[node2][node1] = 0
#             count_negative += 1
#             have_already_sampled[node1][node2] = 1
#             have_already_sampled[node2][node1] = 1

#     return cre_edge_matrix_input

# def generate_random_subgraph( cRE_edge_matrix_input, radius_inp, rand_inp, neighbor_inp, cre_number_of_nodes, cre_number_of_edges, 
#                              cre_edge_matrix_input, cre_attributes, cre_DNA):

#     radius = radius_inp
#     random_size = rand_inp
#     neighbor_size = neighbor_inp
#     sample_size = random_size + neighbor_size

#     edge_list_node1 = []
#     edge_list_node2 = []
#     center = random.randint(radius, cre_number_of_nodes - radius)

#     output_expected = [-1] * (sample_size ** 2)

#     node_set1 = random.sample(range(center - radius, center + radius), k = neighbor_size)
#     node_set2 = random.sample(list(set(range(cre_number_of_nodes)) - set(node_set1)), k = neighbor_size)
#     node_set = node_set1 + node_set2 

#     index_node = list(range(sample_size))

#     attribute_dataset = []

#     node_part1 = random.sample((range(sample_size)), k = sample_size // 2)
#     node_part2 = list(set((range(sample_size))) - set(node_part1))    

#     for node1 in node_part1:
#         for node2 in node_part2: 
#             if(cre_edge_matrix_input[node_set[node1]][node_set[node2]] == 1):
#                 edge_list_node1.extend([node1, node2])
#                 edge_list_node2.extend([node2, node1])

#         output_expected[node1 * sample_size + node2] = cre_edge_matrix_input[node_set[node1]][node_set[node2]]
#         output_expected[node2 * sample_size + node1] = cre_edge_matrix_input[node_set[node2]][node_set[node1]]
    
#     for indx_node in range(sample_size):
#         attribute_dataset.append(cre_attributes[node_set[indx_node]].tolist() + cre_DNA[node_set[indx_node]].tolist())

#     edge_index = torch.tensor([edge_list_node1,edge_list_node2], dtype=torch.long)

#     return Data(x = torch.tensor(np.array(attribute_dataset)), edge_index = torch.tensor(np.array(edge_index)).long(),
#                             y = torch.tensor(np.array(output_expected)).long())


# def generate_subgraphs(cre_dataset, cRE_edge_matrix_input, radius_inp, rand_inp, neighbor_inp, number_samples, ratio_test_size):
 
#   dataset_subgraphs = []

#   for i in tqdm(range(number_samples)):
#     new_subgraph_dataset = generate_random_subgraph(cre_dataset, cRE_edge_matrix_input, radius_inp, rand_inp, neighbor_inp, )
#     dataset_subgraphs.append(new_subgraph_dataset)
  
# #   dataset_train, dataset_test = train_test_split(dataset_subgraphs, test_size = ratio_test_size, random_state = 42)
#   return dataset_subgraphs

if __name__ == "__main__":
    a, b, c, d = load_graph_data("./../Dataset", "cre")    
    print(a, b, c, d)

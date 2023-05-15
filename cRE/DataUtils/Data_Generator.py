import numpy as np 
import torch
import json 
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import os
import random
from tqdm import tqdm

FEATURE_LIST = [3,5,6,7,8,9,10,11,12,13,14]

def generate_datasets(cre_attributes, cre_dna, cre_edge_matrix,cre_edge_pairs):
    cre_number_of_nodes = cre_attributes.size()[0]
    cre_number_of_edges = len(cre_edge_pairs)
    config_file = open('../DataUtils/config.json')
    config_json = json.load(config_file)

    if(config_json["negative_sampling"]):
        cre_edge_matrix = add_negative_samples(cre_number_of_nodes, cre_edge_matrix,config_json["portion_hidden_edges"], cre_number_of_edges)
   
    
    dataset_subgraphs = generate_subgraphs( cre_edge_matrix, cre_attributes, cre_dna,cre_number_of_nodes, config_json["radius"], config_json["random_size"],
                            config_json["neighbor_inp"],config_json["number_samples"])
    
   
    
    feature = torch.tensor(FEATURE_LIST)
    all_features_index = torch.cat([feature, torch.Tensor(list(range(11,779))).long()],dim=0)
    
    dataset_train, dataset_test = train_test_split(dataset_subgraphs , test_size = config_json["ratio_test_size"], random_state = 42)
    dataset_train, dataset_val = train_test_split(dataset_train, test_size = config_json["ratio_test_size"], random_state = 42)
    dataset_train_selected_att = [Data(x = torch.index_select(data.x, 1, all_features_index), edge_index = data.edge_index, y = data.y) for data in  dataset_train]
    dataset_test_selected_att = [Data(x = torch.index_select(data.x, 1, all_features_index), edge_index = data.edge_index, y = data.y) for data in  dataset_test]
    dataset_val_selected_att = [Data(x = torch.index_select(data.x, 1, all_features_index), edge_index = data.edge_index, y = data.y) for data in  dataset_val]
    
    train_loader = DataLoader(dataset_train_selected_att, batch_size = config_json["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset_test_selected_att, batch_size = config_json["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset_val_selected_att, batch_size = config_json["batch_size"], shuffle=True)

    if not os.path.exists(config_json["dataset"]):
        os.makedirs(config_json["dataset"])

    torch.save(train_loader, os.path.join(config_json["dataset"],'train_loader.pth'))
    torch.save(test_loader, os.path.join(config_json["dataset"],'test_loader.pth'))
    torch.save(val_loader, os.path.join(config_json["dataset"],'val_loader.pth'))





def add_negative_samples(cre_number_of_nodes, cre_edge_matrix,portion_negative_to_positive, cre_number_of_edges): 
  
   
    cre_edge_matrix_input =  torch.clone(cre_edge_matrix)

    number_negative_samples = cre_number_of_edges * portion_negative_to_positive
    count_negative = 0

    have_already_sampled = torch.eye(cre_number_of_nodes, cre_number_of_nodes)

    num_node_zero_base = cre_number_of_nodes - 1

    while(count_negative < number_negative_samples):
        
        edge_index_in_matrix = random.randint(0, num_node_zero_base * num_node_zero_base)
        node1 = edge_index_in_matrix // num_node_zero_base
        node2 = edge_index_in_matrix % num_node_zero_base
        if(cre_edge_matrix_input[node1][node2] == -1 and have_already_sampled[node1][node2] == 0): 

            cre_edge_matrix_input[node1][node2] = 0
            cre_edge_matrix_input[node2][node1] = 0
            count_negative += 1
            have_already_sampled[node1][node2] = 1
            have_already_sampled[node2][node1] = 1

    return cre_edge_matrix_input


def generate_random_subgraph( cre_edge_matrix_input, cre_attributes, cre_DNA, cre_number_of_nodes,radius_inp, rand_inp, neighbor_inp):

    radius = radius_inp
    random_size = rand_inp
    neighbor_size = neighbor_inp
    sample_size = random_size + neighbor_size

    edge_list_node1 = []
    edge_list_node2 = []
    center = random.randint(radius, cre_number_of_nodes - radius)

    output_expected = [-1] * (sample_size ** 2)

    node_set1 = random.sample(range(center - radius, center + radius), k = neighbor_size)
    node_set2 = random.sample(list(set(range(cre_number_of_nodes)) - set(node_set1)), k = neighbor_size)
    node_set = node_set1 + node_set2 

    index_node = list(range(sample_size))

    attribute_dataset = []

    node_part1 = random.sample((range(sample_size)), k = sample_size // 2)
    node_part2 = list(set((range(sample_size))) - set(node_part1))    

    for node1 in node_part1:
        for node2 in node_part2: 
            if(cre_edge_matrix_input[node_set[node1]][node_set[node2]] == 1):
                edge_list_node1.extend([node1, node2])
                edge_list_node2.extend([node2, node1])

        output_expected[node1 * sample_size + node2] = cre_edge_matrix_input[node_set[node1]][node_set[node2]]
        output_expected[node2 * sample_size + node1] = cre_edge_matrix_input[node_set[node2]][node_set[node1]]
    
    for indx_node in range(sample_size):
        attribute_dataset.append(cre_attributes[node_set[indx_node]].tolist() + cre_DNA[node_set[indx_node]].tolist())

    edge_index = torch.tensor([edge_list_node1,edge_list_node2], dtype=torch.long)

    return Data(x = torch.tensor(np.array(attribute_dataset)), edge_index = torch.tensor(np.array(edge_index)).long(),
                            y = torch.tensor(np.array(output_expected)).long())


def generate_subgraphs( cre_edge_matrix_input,cre_attributes, cre_DNA,cre_number_of_nodes, radius_inp, rand_inp, neighbor_inp, number_samples):
 
  dataset_subgraphs = []

  for i in tqdm(range(number_samples)):
    new_subgraph_dataset = generate_random_subgraph(cre_edge_matrix_input, cre_attributes, cre_DNA,cre_number_of_nodes ,radius_inp, rand_inp, neighbor_inp)
    dataset_subgraphs.append(new_subgraph_dataset)
  
#   
  return dataset_subgraphs


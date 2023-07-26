import pandas as pd
import cooler
from cooler import Cooler
import numpy as np
import torch
from torch_geometric.data import Data
import os
import sys
import torch_geometric

chunksize=200        
chunk_stride=100   
train_chrom = ["chr" + str(var) for var in range(1,23)]
train_chrom.remove('chr10')
train_chrom.remove('chr15')
val_chrom = ["chr10"]
test_chrom = ["chr15"]
resolution = 10000
data_path = "./"
directory_inp = "../../../dataset"
directory_out = "../ModelDataFiles"
matrix_path = directory_inp + "/H1_HiC.mcool"
save_prefix = "H1HiC90"
def fetch_interaction_data(c1,df_bin1):
    """
    Fetch interaction data from an .mcool file using two dataframes.

    Args:
        filepath (str): Path to the .mcool file.
        df_bin1 (DataFrame): First DataFrame with ['chrom', 'start', 'end'] columns.
        df_bin2 (DataFrame): Second DataFrame with ['chrom', 'start', 'end'] columns.

    Returns:
        List[List]: List of lists, each containing indices of two regions and interaction data for the regions.
    """
    
    interaction_data = []
    start = df_bin1['start'][0]
    end = df_bin1['end'][df_bin1.shape[0]-1]
    matrix = c1.matrix(balance=False).fetch((df_bin1.loc[0, 'chrom'],start, end),(df_bin1.loc[0, 'chrom'],start, end))
    edge=matrix
#     print(edge)
    y=c1.matrix(balance=False).fetch((df_bin1.loc[0, 'chrom'],start, end),(df_bin1.loc[0, 'chrom'],start, end))
    y=np.log(y+1)
    indices_matrix = np.where(edge > 0)
    values_edge = edge[indices_matrix]
    index1_edge = indices_matrix[0].flatten()
    index2_edge = indices_matrix[1].flatten()
    values_edge = np.log(values_edge.flatten()+1)
    stacked_array=np.column_stack((index1_edge, index2_edge, values_edge))
    
    num_rows = stacked_array.shape[0]
    # Calculate the number of rows to select (90% of the total)
    num_rows_to_select = int(0.9 * num_rows)
    # Generate random indices to select
    random_indices = np.random.choice(num_rows, size=num_rows_to_select, replace=False)
    # Select the rows using the random indices
    interaction_data = stacked_array[random_indices]

    return interaction_data , y

def check_continuity(df):
    # Create shifted columns for 'start', 'end', and 'chromosome'
    df['next_start'] = df['start'].shift(-1)
    df['next_chromosome'] = df['chrom'].shift(-1)

    df['is_continuous'] = (df['end'] == df['next_start'])
    df = df.drop(df[df['chrom'] != df['next_chromosome']].index)

    return df['is_continuous'].all()

def create_dataset(chrom_list,data_path,matrix_path):
    dataset=[]
    for chrom in chrom_list:
        print(f"assembling {chrom}...")
        bedfile = os.path.join(data_path,directory_inp + "/10kb_bedfiles" + f"/{chrom}_{resolution}bins.bed")
        Bins = pd.read_table(bedfile, sep='\t', header=None, names=['chrom', 'start', 'end'])
        attributes = np.load(os.path.join(data_path, directory_inp + f"/attributes/{chrom}_attributes_10kb.npy"))
        Encoded_DNA = np.load(os.path.join(data_path, directory_inp + f"/DNA/{chrom}.npy"))
        matrix = Cooler(matrix_path+"::/resolutions/10000")  

        chunklist_length = int(Bins.shape[0] / chunk_stride)


        for i in range(chunklist_length):
            chunkstart = i * chunk_stride
            local_bin = []
            if (chunkstart + chunksize) > Bins.shape[0]:
                break

            chunk = Bins.iloc[chunkstart:chunkstart + chunksize].reset_index(drop=True)
            if not check_continuity(chunk):
                continue

            interaction_data, y = fetch_interaction_data(matrix, chunk)
            edge_index = torch.tensor([item[:2] for item in interaction_data]).long().t()
            edge_attr = torch.tensor([item[2] for item in interaction_data], dtype=torch.float)

            genomicfeature = attributes[chunkstart:chunkstart + chunksize, :]
            DNAfeature = Encoded_DNA[chunkstart:chunkstart + chunksize, :]
            x_dataset = np.concatenate((genomicfeature, DNAfeature), axis=1)

            dataset.append(torch_geometric.data.Data(x=torch.tensor(np.array(x_dataset)),
                                                     edge_index=edge_index,
                                                     edge_attr=edge_attr, y=torch.tensor(np.array(y))))

    return dataset

dataset = create_dataset(train_chrom,data_path,matrix_path)
torch.save(dataset,directory_out + f"/{save_prefix}_train.pt")
dataset = create_dataset(val_chrom,data_path,matrix_path)
torch.save(dataset,directory_out + f"/{save_prefix}_val.pt")
dataset = create_dataset(test_chrom,data_path,matrix_path)
torch.save(dataset,directory_out + f"/{save_prefix}_test.pt")

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import random
import sys
import copy
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from tqdm import tqdm
import json
from itertools import combinations
from torch_geometric.loader import DataLoader
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, pearsonr

# cRE_att = np.load('chr1_attributes_10kb.npy')
# suffix='PancHiC'
suffix='H1HiC'
dataset_train=torch.load(f"{suffix}_train.pt")
dataset_val=torch.load(f"{suffix}_val.pt")
feature_num=10
# dataset_train=torch.load("H1chr1-train_1set_balance.pt")

#dataset_test=torch.load("/home/ruoyun/pyworking/binnedgenome/H1chr1-test.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

batchSize=14
chunksize=200        
samplesize=100 
sample_stride=50
chunk_stride=100   
gcnInputsize=768+feature_num
gcnHiddensize=400
gcnOutputsize=768


import copy
class network(torch.nn.Module):
    def __init__(self,indexsize,gcnInputsize,gcnHiddensize,gcnOutputsize,batchSize):
        super(network, self).__init__()
        
        self.conv1 = GCNConv(gcnInputsize, gcnHiddensize)
        self.conv1_2 = GCNConv(gcnHiddensize, 2 * gcnHiddensize)
        self.conv2 = GCNConv(2 * gcnHiddensize, gcnOutputsize)
        self.transform_conv1 = nn.Conv1d(in_channels=2*gcnOutputsize, out_channels=int(1.5*gcnOutputsize), kernel_size=3, stride=1,padding=1)
        self.transform_conv2 = nn.Conv1d(in_channels=int(1.5*gcnOutputsize), out_channels=gcnOutputsize, kernel_size=3, stride=1,padding=1)
        self.transform_conv3 = nn.Conv1d(in_channels=gcnOutputsize, out_channels=gcnOutputsize//2, kernel_size=3, stride=1,padding=1)
        self.transform_conv4 = nn.Conv1d(in_channels=gcnOutputsize//2, out_channels=gcnOutputsize//4, kernel_size=3, stride=1,padding=1)
        self.transform_conv5 = nn.Conv1d(in_channels=gcnOutputsize//4, out_channels=gcnOutputsize//8, kernel_size=3, stride=1,padding=1)
        self.transform_conv6 = nn.Conv1d(in_channels=gcnOutputsize//8, out_channels=gcnOutputsize//16, kernel_size=3, stride=1,padding=1)
        self.transform_conv7 = nn.Conv1d(in_channels=gcnOutputsize//16, out_channels=gcnOutputsize//32, kernel_size=3, stride=1,padding=1)
        self.transform_conv8 = nn.Conv1d(in_channels=gcnOutputsize//32, out_channels=1, kernel_size=3, stride=1,padding=1)

        self.linear1=torch.nn.Linear(2*gcnOutputsize,gcnOutputsize)
        self.linear2 = torch.nn.Linear(gcnOutputsize, 1)

        self.indexsize=indexsize       
        self.batchsize=batchSize
        self.bn = nn.BatchNorm1d(gcnOutputsize, affine=False)  # BatchNorm1d layer
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=gcnOutputsize, nhead=12, batch_first=True)
        self.transformer_layers = [copy.deepcopy(self.transformer_layer).to(device) for i in range(12)]

    def forward(self, data,batchSize=batchSize):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr

        x = self.conv1(x.float(), edge_index,edge_attr)
        x = F.relu(x)
        x = self.conv1_2(x, edge_index,edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index,edge_attr)

        x = x.view(batchSize, 200, -1)
        for layers in self.transformer_layers:
            x1 = layers(x)            
            x1 = x1.permute(0, 2, 1).contiguous()
            x1 = self.bn(x1)
            x1 = x1.permute(0, 2, 1).contiguous()
            x1 = F.relu(x1)
            x += x1
        x = F.dropout(x, training=self.training)
        origs = x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1), -1).reshape(x.size(0), x.size(1) * x.size(1), -1)
        dests = x.unsqueeze(1).expand(x.size(0), x.size(1), x.size(1), -1).reshape(x.size(0), x.size(1) * x.size(1), -1)
        embs = torch.cat([origs, dests], dim=2)

        embs = embs.permute(0, 2, 1)
        x = F.relu(self.transform_conv1(embs))
        x = F.relu(self.transform_conv2(x))
        x = F.relu(self.transform_conv3(x))
        x = F.relu(self.transform_conv4(x))
        x = F.relu(self.transform_conv5(x))
        x = F.relu(self.transform_conv6(x))
        x = F.relu(self.transform_conv7(x))
        x = self.transform_conv8(x)

        x = x.permute(0, 2, 1)

        return x.view(-1)


model=network(feature_num,gcnInputsize,gcnHiddensize,gcnOutputsize,batchSize)
model.to(device) 

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


train_loader = DataLoader(dataset_train, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
val_list=[]
train_list=[]
F1_list=[]
pearson_list=[]
spear_list=[]
patience = 100
min_val_loss = float('inf')
counter = 0
best_val=0
for epoch in tqdm(range(800)):
    count=0
    loss=0.0

    model.train()
    trainloss=0.
    for data in train_loader:
        data = data.to(device)
        if data.x.size(0) < 2800:
            break
        pred_y = model(data)

        mask = ~torch.isnan(data.y.view(-1))

        non_nan_indexes = torch.nonzero(mask).squeeze()
        labels = torch.index_select(data.y.view(-1), 0, non_nan_indexes)
        pred_y = torch.index_select(pred_y, 0, non_nan_indexes)
        loss = loss + criterion(pred_y, labels.float())
        trainloss+=loss.item()
        count=count+1
        if(count==2):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss=0.0
            klloss = 0.
            count=0 

        del(labels)
        del(non_nan_indexes)
    if(loss!=0):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('epoch {},Train loss {}'.format(epoch, trainloss/len(train_loader)))

    model.eval()
    val_loss = 0
    ytrue = []
    ypred = []
    pearsoncorr = 0
    spearmancorr = 0
    if epoch % 5 == 0:
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                pred_y = model(data,batchSize=1)
                mask = ~torch.isnan(data.y.view(-1))
                non_nan_indexes = torch.nonzero(mask).squeeze()
                labels = torch.index_select(data.y.view(-1), 0, non_nan_indexes)
                pred_y_tmp = torch.index_select(pred_y, 0, non_nan_indexes)
                ytrue += labels.cpu().tolist()
                ypred += pred_y_tmp.cpu().tolist()
                loss = criterion(pred_y_tmp, labels.float())


                pearsoncorr += pearsonr(ytrue, ypred)[0]
                spearmancorr += spearmanr(ytrue, ypred)[0]

                del(labels)
                del(non_nan_indexes)
                val_loss += loss.item()
        val_loss/=len(val_loader)
        print('Epoch: {}, Val Loss: {:.4f}'.format(epoch, val_loss))
        print('Val corrs', pearsoncorr/len(val_loader), spearmancorr/len(val_loader))
        pearson_list.append(pearsoncorr/len(val_loader))
        spear_list.append(spearmancorr/len(val_loader))
        if pearsoncorr > best_val:
            best_val = pearsoncorr
            counter = 0
            # Save the best model
            print(f"Saved the best model.")
            torch.save(model,f"bestmodel_{suffix}.pt")
            torch.save(torch.tensor(val_list),f"valoss_{suffix}_best.pt")
            torch.save(torch.tensor(train_list),f"trainloss_{suffix}_best.pt")
            torch.save(torch.tensor(pearson_list),f"pearson_{suffix}_best.pt")
            torch.save(torch.tensor(spear_list),f"spear_{suffix}_best.pt")

        else:
            counter += 1
            print(f"Early stop counter {counter}")


    train_list.append(trainloss/len(train_loader))
    val_loss /= len(val_loader)
    val_list.append(val_loss)

torch.save(model,f"model_{suffix}.pt")
torch.save(torch.tensor(val_list),f"valoss_{suffix}.pt")
torch.save(torch.tensor(train_list),f"trainloss_{suffix}.pt")
torch.save(torch.tensor(pearson_list),f"pearson_{suffix}.pt")
torch.save(torch.tensor(spear_list),f"spear_{suffix}.pt")

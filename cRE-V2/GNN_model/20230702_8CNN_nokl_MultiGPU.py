
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
from torch_geometric.nn import DataParallel, GCNConv

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch.optim as optim


from torch.cuda.amp import autocast, GradScaler


import torch.cuda.amp as amp



from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, pearsonr

from datetime import datetime
import logging

# Get the current time
current_time = datetime.now()

# Format the current time as a string without spaces
formatted_time = current_time.strftime("%m%d%H%M%S")


# Configure the logging settings

directory_inp = "../ModelDataFiles/new_dataset/"
datasuffix = 'H1HiC90'

# dataset_train = torch.load(directory_inp + f"{suffix}_train.pt")
# dataset_val = torch.load(directory_inp + f"{suffix}_val.pt")

logfolder = "./log/"
savefolder = "./results/"
if not os.path.exists(logfolder):
    # If it doesn't exist, create the folder
    os.makedirs(logfolder)
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
savingsuffix=f'H1microC90_s50_CE_MSE_nobnrs_gelu_{formatted_time}'
logging.basicConfig(
    filename=f'./log/{savingsuffix}.log',  # Specify the log file name
    level=logging.INFO,          # Set the log level (you can use DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

dataset_train = torch.load(directory_inp + f"{datasuffix}_train.pt")
dataset_val = torch.load(directory_inp + f"{datasuffix}_val.pt")
feature_num = 6


batchSize = 10
chunksize = 200        
samplesize = 200 
sample_stride = 50
chunk_stride = 100   
gcnInputsize = 768 + feature_num
gcnHiddensize = 400
gcnOutputsize = 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def signal_handler(signal, frame):
    # Perform necessary cleanup here, such as closing files or connections.
    # Then exit the process.
    sys.exit(0)

import copy
class network(torch.nn.Module):
    def __init__(self,indexsize,gcnInputsize,gcnHiddensize,gcnOutputsize,batchSize):
        super(network, self).__init__()
        self.conv1 = GCNConv(gcnInputsize, gcnHiddensize)
        self.conv1_2 = GCNConv(gcnHiddensize, 2 * gcnHiddensize)
        self.conv2 = GCNConv(2 * gcnHiddensize, gcnOutputsize)
        # self.conv2_2 = GCNConv(3 * gcnHiddensize, gcnOutputsize)
        self.transform_conv1 = nn.Conv1d(in_channels=2*gcnOutputsize, out_channels=int(1.5*gcnOutputsize), kernel_size=3, stride=1,padding=1)
        self.transform_conv2 = nn.Conv1d(in_channels=int(1.5*gcnOutputsize), out_channels=gcnOutputsize, kernel_size=3, stride=1,padding=1)
        self.transform_conv3 = nn.Conv1d(in_channels=gcnOutputsize, out_channels=gcnOutputsize//2, kernel_size=3, stride=1,padding=1)
        self.transform_conv4 = nn.Conv1d(in_channels=gcnOutputsize//2, out_channels=gcnOutputsize//4, kernel_size=3, stride=1,padding=1)
        self.transform_conv5 = nn.Conv1d(in_channels=gcnOutputsize//4, out_channels=gcnOutputsize//8, kernel_size=3, stride=1,padding=1)
        self.transform_conv6 = nn.Conv1d(in_channels=gcnOutputsize//8, out_channels=gcnOutputsize//16, kernel_size=3, stride=1,padding=1)
        self.transform_conv7 = nn.Conv1d(in_channels=gcnOutputsize//16, out_channels=gcnOutputsize//32, kernel_size=3, stride=1,padding=1)
        self.transform_conv8 = nn.Conv1d(in_channels=gcnOutputsize//32, out_channels=1, kernel_size=3, stride=1,padding=1)
        self.cls_head1 = nn.Linear(2*gcnOutputsize, gcnOutputsize)
        self.cls_head2 = nn.Linear(gcnOutputsize, gcnOutputsize//2)
        self.cls_head3 = nn.Linear(gcnOutputsize//2, gcnOutputsize//16)
        self.cls_head4 = nn.Linear(gcnOutputsize//16, 8)

        # = nn.Conv1d(in_channels=gcnOutputsize//32, out_channels=3, kernel_size=3, stride=1,padding=1)
        # self.matrix=Variable(torch.randn(gcnOutputsize,gcnOutputsize).type(torch.FloatTensor),requires_grad=True)
        self.linear1=torch.nn.Linear(2*gcnOutputsize,gcnOutputsize)
        self.linear2 = torch.nn.Linear(gcnOutputsize, 1)
#         self.top_transformer_layer = copy.deepcopy(self.transformer_layer).to(device)
        # self.transform1=nn.Linear(indexsize, gcnInputsize)
        # self.transform2=nn.Linear(768,gcnInputsize)
        self.indexsize=indexsize       
        self.batchsize=batchSize
        self.bn = nn.BatchNorm1d(gcnOutputsize, affine=False)  # BatchNorm1d layer
        # upper = [0] * samplesize + [1] * samplesize
        # lower = [1] * samplesize + [0] * samplesize
        # self.attention_mask = torch.tensor([upper] * samplesize + [lower] * samplesize).float().to(device)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=gcnOutputsize, nhead=12, batch_first=True)
        self.transformer_layers = [copy.deepcopy(self.transformer_layer).to(device) for i in range(12)]

    def forward(self, data,batchSize=batchSize):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr
        # x1=x[:,:self.indexsize].float()
        # x2=x[:,self.indexsize:].float()
        # x = self.transform1(x1)+self.transform2(x2)
        x = self.conv1(x.float(), edge_index,edge_attr)
        x = F.gelu(x)
        x = self.conv1_2(x, edge_index,edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index,edge_attr)
        # x = F.relu(x)
        # x = self.conv2_2(x, edge_index)
        # with open("output.txt", "a") as file:
        #     print(x.size(), file=file)
        x = x.view(batchSize, 200, -1)
        for layers in self.transformer_layers:
            x1 = layers(x)            
            # x1 = x1.permute(0, 2, 1).contiguous()
            # x1 = self.bn(x1)
            # x1 = x1.permute(0, 2, 1).contiguous()
            x1 = F.gelu(x1)
            # x = x + x1
            x = x1

        x = F.dropout(x, training=self.training)
        # embs = []
        origs = x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1), -1).reshape(x.size(0), x.size(1) * x.size(1), -1)
        dests = x.unsqueeze(1).expand(x.size(0), x.size(1), x.size(1), -1).reshape(x.size(0), x.size(1) * x.size(1), -1)
        embs = torch.cat([origs, dests], dim=2)
        # print(embs.size())
        # for i in range(x.size(1)):
        #     for j in range(x.size(1)):
        #         emb = torch.cat([x[:,i,:].unsqueeze(1), x[:,j,:].unsqueeze(1)], dim=2)
        #         embs.append(emb)
        # embs = torch.cat(embs, dim=1)
        # print(embs.size(), gcnOutputsize)
        cls_pred = F.tanh(self.cls_head1(embs))
        cls_pred = F.relu(self.cls_head2(cls_pred))
        cls_pred = F.tanh(self.cls_head3(cls_pred))
        cls_pred = self.cls_head4(cls_pred)
        embs = embs.permute(0, 2, 1).contiguous()
        x = F.gelu(self.transform_conv1(embs))
        x = F.gelu(self.transform_conv2(x))
        x = F.gelu(self.transform_conv3(x))
        x = F.gelu(self.transform_conv4(x))
        x = F.gelu(self.transform_conv5(x))
        x = F.gelu(self.transform_conv6(x))
        x = F.gelu(self.transform_conv7(x))
        x = self.transform_conv8(x)
        x = x.permute(0, 2, 1).contiguous()

        # x = F.tanh(x)
        # x = self.linear2(x)
#         x = self.top_transformer_layer.self_attn(x,x,x)[1]
# #         x = x.view(-1, 100, 100)
#         #print(x)
#         x = self.linear(x.view(-1,1))
        # print(x.size())
        # print(cls_pred.size())  
        return x.view(-1), cls_pred.reshape( -1, 8)


def main(rank, world_size):

    # torch.backends.cudnn.benchmark = True
    scaler = amp.GradScaler()

    dist.init_process_group(backend='nccl')

    device = torch.device('cuda',rank) 
    model = network(feature_num,gcnInputsize,gcnHiddensize,gcnOutputsize,batchSize)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    torch.autograd.set_detect_anomaly(True)
    KLcriterion = torch.nn.KLDivLoss()
    hingeCriterion = torch.nn.MultiMarginLoss()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    crossEntropyCriterion = torch.nn.CrossEntropyLoss()
    MAEcriterion = torch.nn.L1Loss()

    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank = rank)
    sampler_val = DistributedSampler(dataset_val, num_replicas=world_size, rank = rank)

    train_loader = DataLoader(dataset_train, batch_size = batchSize,shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size = 1)

    val_list = []
    train_list = []
    F1_list = []
    pearson_list = []
    spear_list = []
    patience = 100
    min_val_loss = float('inf')
    counter = 0
    best_val = 0
    for epoch in tqdm(range(800)):
        count = 0
        loss = 0.0
        model.train()
        trainloss = 0.
        CEloss=0
        # MAEloss=0
        for data in (train_loader):
            data = data.to(device)
            if data.x.size(0) < batchSize*samplesize:
                break
            with autocast():
                pred_y,pred_yc= model(data)
                if epoch % 200 <100 :
                    labels_C = data.yc.view(-1)
                    # labels = data.y.view(-1)
                    # print(F.log_softmax(pred_yc.view(batchSize,-1,8), dim=2))
                    # print("pred y",pred_y,"\n pred_yc ",pred_yc)
                    # print("data.yc.view(-1) ",data.yc.view(-1),"\ndata.y.view(-1)",data.y.view(-1))

                    loss = loss + crossEntropyCriterion(pred_yc, labels_C) 
                   # print("loss is ",loss)

                 #   loss = loss + criterion(pred_y, data.y.view(-1).float()) +KLcriterion(pred_y,data.y.view(-1).float())
            

                else:
                    labels = data.y.view(-1)
                    loss = loss + criterion(pred_y, labels.float()) + KLcriterion(pred_y, labels.float())             

                trainloss+=loss.item()
                count=count+1
                if(count == 2):
                    scaler.scale(loss).backward()
                    max_norm = 1  # You can set your desired max norm value
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    loss = 0.0
                    # CEloss = 0.
                    count = 0 

        if(loss!=0):
            scaler.scale(loss).backward()
            max_norm = 1  # You can set your desired max norm value
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        print('epoch {}, Train loss {}'.format(epoch, trainloss / len(train_loader)))
        logging.info('epoch {}/800, Train loss {}'.format(epoch, trainloss / len(train_loader)))
        model.eval()
        val_loss = 0
        ytrue = []
        ypred = []
        pearsoncorr = 0
        spearmancorr = 0
        val_CEloss = 0
        val_MAEloss = 0
        if epoch % 5 == 0:
            with torch.no_grad():
                for data in (val_loader):
                    data = data.to(device)
                    with autocast():
                        pred_y ,pred_yc = model(data,batchSize = 1)
                        # mask = ~torch.isnan(data.y.view(-1))
                        # non_nan_indexes = torch.nonzero(mask).squeeze()
                        # labels = torch.index_select(data.y.view(-1), 0, non_nan_indexes)
                        # pred_y_tmp = torch.index_select(pred_y, 0, non_nan_indexes)
                        labels = data.y.view(-1)
                        ytrue += labels.cpu().tolist()
                        ypred += pred_y.cpu().tolist()
                        loss = criterion(pred_y, labels.float())
                        labels_C = data.yc.view(-1)
                        CEloss = hingeCriterion(pred_yc, labels_C)
                        MAEloss = MAEcriterion(pred_y, labels.float())
                        pearsoncorr += pearsonr(ytrue, ypred)[0]
                        spearmancorr += spearmanr(ytrue, ypred)[0]
                        val_loss += loss.item()
                        val_CEloss += CEloss.item()
                        val_MAEloss += MAEloss.item()
            val_loss/=len(val_loader)
            val_CEloss/=len(val_loader)
            pearsoncorr/=len(val_loader)
            spearmancorr/=len(val_loader)
            val_MAEloss/=len(val_loader)
            print('Epoch: {}, Val MSELoss: {:.4f}, Val HGloss:{:.4f}, Val MAEloss:{:.4f}'.format(epoch, val_loss, val_CEloss, val_MAEloss))
            print('Val corrs', pearsoncorr, spearmancorr)
            logging.info('Epoch: {}, Val MSELoss: {:.4f}, Val HGloss:{:.4f}, Val MAEloss:{:.4f}'.format(epoch, val_loss, val_CEloss, val_MAEloss))
            logging.info(f'Val corrs pearson:{pearsoncorr}, spearman:{spearmancorr}')
            pearson_list.append(pearsoncorr)
            spear_list.append(spearmancorr)
            if pearsoncorr > best_val:
                best_val = pearsoncorr
                counter = 0
                # Save the best model
                print(f"Saved the best model as bestmodel_{savingsuffix}.pt.")

                torch.save(model, f"{savefolder}bestmodel_{savingsuffix}.pt")
                torch.save(torch.tensor(val_list), f"{savefolder}valoss_{savingsuffix}_best.pt")
                torch.save(torch.tensor(train_list), f"{savefolder}trainloss_{savingsuffix}_best.pt")
                torch.save(torch.tensor(pearson_list), f"{savefolder}pearson_{savingsuffix}_best.pt")
                torch.save(torch.tensor(spear_list), f"{savefolder}spear_{savingsuffix}_best.pt")
                logging.info(f"Saved the best model as bestmodel_{savingsuffix}.pt.")
            else:
                counter += 1
                print(f"Early stop counter {counter}. Best val:{best_val} saved as bestmodel_{savingsuffix}.pt")
                logging.info(f"Early stop counter {counter}. Best val:{best_val} saved as bestmodel_{savingsuffix}.pt")

        train_list.append(trainloss / len(train_loader))
        val_list.append(val_loss)

    torch.save(model, f"{savefolder}model_{savingsuffix}.pt")
    torch.save(torch.tensor(val_list), f"{savefolder}valoss_{savingsuffix}.pt")
    torch.save(torch.tensor(train_list), f"{savefolder}trainloss_{savingsuffix}.pt")
    torch.save(torch.tensor(pearson_list), f"{savefolder}pearson_{savingsuffix}.pt")
    torch.save(torch.tensor(spear_list), f"{savefolder}spear_{savingsuffix}.pt")
    logging.info(f"final model saved as model_{savingsuffix}.pt")
    # dist.destroy_process_group()

if __name__ == '__main__':

   # main()
    logging.info('Process completed successfully.')
    mp.spawn(main, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(),))

    # Set up the multiprocessing environment

    # mp.spawn(main, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(),))



# directory_inp = "../ModelDataFiles/"
# suffix = 'H1HiC90'
# dataset_train = torch.load(directory_inp + f"{suffix}_train.pt")
# dataset_val = torch.load(directory_inp + f"{suffix}_val.pt")
# suffix = 'H1HiC90_KLL'
# feature_num = 10


# batchSize = 15
# chunksize = 200        
# samplesize = 100 
# sample_stride = 50
# chunk_stride = 100   
# gcnInputsize = 768 + feature_num
# gcnHiddensize = 400
# gcnOutputsize = 768

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--epoch",
#         required=False,
#         default=100,
#         help="Number of epochs",
#     )   
#     parser.add_argument(
#         "--batch_size",
#         required=False,
#         default=5,
#         help="batch size",
#     )   
#     return parser.parse_args()

# def signal_handler(signal, frame):
#     # Perform necessary cleanup here, such as closing files or connections.
#     # Then exit the process.
#     sys.exit(0)



# def main(rank, world_size,epochs, batch_size):
#    # torch.cuda.empty_cache()
#     global batchSizes
#     batchSize = batch_size
    
    
#     scaler = amp.GradScaler()   

#     dist.init_process_group(backend='nccl')
#    # print(f"epcohs: {epoch} -- batch size: {batch_size}")
   
#     device = torch.device('cuda', rank) 
    
    
#     torch.autograd.set_detect_anomaly(True)
#     model = network(feature_num,gcnInputsize,gcnHiddensize,gcnOutputsize,batchSize)
#     model.to(device)
    
#     model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)


#     criterion_mse = torch.nn.MSELoss()
#     criterion_kll = torch.nn.KLDivLoss().to(device) 
#     optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    
#     sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank = rank)
#     sampler_val = DistributedSampler(dataset_val, num_replicas=world_size, rank = rank)

#     train_loader = DataLoader(dataset_train, batch_size = batchSize, sampler = sampler_train)
#     val_loader = DataLoader(dataset_val, batch_size = 1, sampler = sampler_val)

#     val_list = []
#     train_list = []
#     F1_list = []
#     pearson_list = []
#     spear_list = []
#     patience = 100
#     min_val_loss = float('inf')
#     counter = 0
#     best_val = 0
#     for epoch in tqdm(range(100)):
       
#         count = 0
#         loss = 0.0
#         model.train()
#         trainloss = 0.
#         for data in tqdm(train_loader):
#             data = data.to(rank)
#             x = data.x
#             dummy_elements = np.zeros((x.shape[0], gcnInputsize - x.shape[1]))
#             dummy_elements = torch.tensor(dummy_elements).to(x.device)
        
#             x = torch.cat((x, dummy_elements), dim = 1)
#             x = x.float()
#             data.y = data.y.float()
#             # data.edge_attr = data.edge_attr.float()
            
#             data.x = x
#            # print("data.x.size(0) ",data.x.size(0))
#             if data.x.size(0) < 3500:
#                 break
#             data = data.to(rank)
#             with amp.autocast(dtype=torch.float16):
               
                
#                 pred_y = model(data)

#                 mask = ~torch.isnan(data.y.view(-1))
#                 non_nan_indexes = torch.nonzero(mask).squeeze()
#                 labels = torch.index_select(data.y.view(-1), 0, non_nan_indexes)
#                 pred_y = torch.index_select(pred_y, 0, non_nan_indexes)
                
                
#                 loss = loss + criterion_kll(pred_y, labels) + criterion_mse(pred_y,labels)
               
       

#             trainloss += loss.item()
            
#             count = count + 1

#             if(count == 2):
                
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)

#                 scaler.update()

#                 loss = 0.0
#                 klloss = 0.
#                 count = 0 

#                 del(labels)
#                 del(non_nan_indexes)
            
#         if(loss != 0):

            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)

#             scaler.update()
#             optimizer.zero_grad()
#             loss = 0.0
#             klloss = 0.
#             count = 0 
      
#         print('epoch {}, Train loss {}'.format(epoch, trainloss / len(train_loader)))

#         model.eval()
#         val_loss = 0
#         ytrue = []
#         ypred = []
#         pearsoncorr = 0
#         spearmancorr = 0
        
#         if epoch % 5 == 0:
#             with torch.no_grad():
#                 for data in tqdm(val_loader):
                   
#                     data = data.to(device)
#                     x = data.x
#                     dummy_elements = np.zeros((x.shape[0], gcnInputsize - x.shape[1]))
#                     dummy_elements = torch.tensor(dummy_elements).to(x.device)

#                     x = torch.cat((x, dummy_elements), dim = 1)
#                     x = x.float()
#                     data.y = data.y.float()
#                     data.x = x
#                     data = data.to(device)

#                     with amp.autocast(dtype=torch.float16):
                    
#                         pred_y = model(data,batchSize = 1)
#                         mask = ~torch.isnan(data.y.view(-1))
#                         non_nan_indexes = torch.nonzero(mask).squeeze()
#                         labels = torch.index_select(data.y.view(-1), 0, non_nan_indexes)
#                         pred_y_tmp = torch.index_select(pred_y, 0, non_nan_indexes)
#                         ytrue += labels.cpu().tolist()
#                         ypred += pred_y_tmp.cpu().tolist()
#                         loss = criterion_kll(pred_y_tmp, labels.float()) + criterion_mse(pred_y_tmp, labels.float())


#                         pearsoncorr += pearsonr(ytrue, ypred)[0]
#                         spearmancorr += spearmanr(ytrue, ypred)[0]

#                     del(labels)
#                     del(non_nan_indexes)
#                     val_loss += loss.item()
#             val_loss /= len(val_loader)
#             print('Epoch: {}, Val Loss: {:.4f}'.format(epoch, val_loss))
#             print('Val corrs', pearsoncorr / len(val_loader), spearmancorr / len(val_loader))
#             pearson_list.append(pearsoncorr / len(val_loader))
#             spear_list.append(spearmancorr / len(val_loader))
#             if pearsoncorr > best_val:
#                 best_val = pearsoncorr
#                 counter = 0
#                 # Save the best model
#                 print(f"Saved the best model.")
#                 torch.save(model, f"bestmodel_{suffix}.pt")
#                 torch.save(torch.tensor(val_list), f"valoss_{suffix}_best.pt")
#                 torch.save(torch.tensor(train_list), f"trainloss_{suffix}_best.pt")
#                 torch.save(torch.tensor(pearson_list), f"pearson_{suffix}_best.pt")
#                 torch.save(torch.tensor(spear_list), f"spear_{suffix}_best.pt")

#             else:
#                 counter += 1
#                 print(f"Early stop counter {counter}")


#         train_list.append(trainloss / len(train_loader))
#         val_loss /= len(val_loader)
#         val_list.append(val_loss)

#     torch.save(model, f"model_{suffix}.pt")
#     torch.save(torch.tensor(val_list), f"valoss_{suffix}.pt")
#     torch.save(torch.tensor(train_list), f"trainloss_{suffix}.pt")
#     torch.save(torch.tensor(pearson_list), f"pearson_{suffix}.pt")
#     torch.save(torch.tensor(spear_list), f"spear_{suffix}.pt")
#     print("perocess finished")
#     dist.destroy_process_group()
   

# if __name__ == '__main__':
#     # Set up the multiprocessing environment
#     #torch.cuda.empty_cache()
    
   
#     args = parse_args()
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
   

#     mp.spawn(main, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(),int(args.epoch),int(args.batch_size),))


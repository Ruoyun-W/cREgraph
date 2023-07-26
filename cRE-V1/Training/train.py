
import torch
from torch_geometric.nn import GCNConv,summary
import json


def train_gnn_model(model,dataset_train, dataset_test, config_path):
  

  config_file = open(config_path)
  config_json = json.load(config_file)

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
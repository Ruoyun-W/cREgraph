
import torch
import numpy as np
import json
from sklearn.metrics import f1_score
from Testing.utils import make_file_adj_graph

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
      #print((data.y == -1).sum())
      if(cnt <= 2):
        mat_y = np.reshape(data.y[0:10000], (100, 100))
        mat_pred_y = np.reshape(y_pred[0:10000], (100, 100))
        mat_y = mat_y.numpy()
        make_file_adj_graph(mat_y, "original_sample_100", "../../Result/sample100_original.png")
        make_file_adj_graph(mat_pred_y, "pred_sample_100", "../../Result/sample100_pred.png")

        

  y_pred_samples = [y_pred[i] for i in range(len(y_pred)) if y_true[i] != -1] 
  y_true_samples = [var for var in y_true if var != -1] 
  f1 = f1_score(y_true_samples, y_pred_samples, average='macro')
  print("f1 score is: ", f1)      

  return y_true, y_pred
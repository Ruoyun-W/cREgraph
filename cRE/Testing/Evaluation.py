def evaluate_result(y_pred, y_true):
  
  true_pos=0
  false_neg=0
  new_pos=0
  new_neg=0
  true_neg=0
  false_pos=0

  for i in range(len(y_pred)):
      if(y_true[i] == 1):
          if(y_pred[i] == 1):
              true_pos += 1
          else:
              false_neg += 1
      elif(y_true[i] == -1):
          if(y_pred[i] == 1):
              new_pos += 1
          else:
              new_neg += 1
      elif(y_true[i] == 0):
          if(y_pred[i] == 1):
              false_pos += 1
          else:
              true_neg += 1
              
  print("True positive is: ", true_pos)
  print("False negative is: ", false_neg)
  print("New positive is: ", new_pos)
  print("New negative is: ", new_neg)
  print("False positive is: ", false_pos)
  print("True negative is: ", true_neg)
  print("Precision(+) is: ", true_pos / (true_pos + false_pos))
  print("Precision(-) is: ", true_neg / (true_neg + false_neg))
  print("Recall(+) is: ", true_pos / (true_pos + false_neg))
  print("Recall(-) is: ", true_neg / (true_neg + false_pos))
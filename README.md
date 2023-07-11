# cREgraph


![visualization of the graph](https://github.com/Ruoyun-W/cREgraph/assets/87666536/005e65f1-2d15-4b73-a4dd-e1da93edecd9)


## Overview 

Here it is an implementation of various models for link prediction in cRE graph with TensorFlow. The repository organization in cRE folder: 

- DataUtils: Contains the python files for generating training, validation and testing dataset which itself contains of number of subgraphs with nodes attributes and edges among them. Also there is an ipynb file for feature extraction and generating files for visualization in gephi.
- Models: contains various models that could have some benefits for prediction of links between cREs.
- Training: consist of training.py for training the model with given dataset.
- Testing: contains test.py for testing the model and calculating f1-score, also the evaluation.py for evaluating the results of the test and finding number of positive-negative prediction and to see how well the model perfomrs.
- Result: All of the results of running the model goes here.

Also there is cRE.ipynb that utilize all the function to run and evaluate a given model.

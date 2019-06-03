'''
Here is a basic script for training and testing random forest and neural network
regression models. The script closely follows the Target Prediction Tutorial
'''

# import packages for uploading data
import pandas as pd
import numpy as np
# import ML_microbiome package
from ML_microbiome import Model, FeatureSelection
import matplotlib.pyplot as plt
from scipy.stats import linregress

# import training and test data as pandas dataframes
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
targets = pd.read_csv('DOC_targets.csv')

taxa = pd.read_csv('selected_taxa.csv')['Taxa'].values

train_data_FS = train_data[taxa]
train_data_FS.insert(0, 'Sample ID', train_data[train_data.columns[0]])

test_data_FS = test_data[taxa]
test_data_FS.insert(0, 'Sample ID', test_data[test_data.columns[0]])

#%% instantiate model class

model = Model('Neural Network', train_data_FS, test_data_FS, targets)
#print(model.tune_hyper_params())

# train model
model.train_model()

# test model
Y_test, Y_pred_test_NN, Y_err_NN = model.test_model()

#%% instantiate model class

model = Model('Random Forest', train_data_FS, test_data_FS, targets)
#print(model.tune_hyper_params())

# train model
model.train_model()

# test model
ytest, ypredRF, rRF = model.test_model()

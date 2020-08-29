import os
import matplotlib.pyplot as plt
# import packages for uploading data
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
# import ML_microbiome package
from RFINN import *

# determine filenames of permutations
train_path = 'train/'
train_perms = sorted(os.listdir(train_path))
test_path = 'test/'
test_perms = sorted(os.listdir(test_path))

targets = pd.read_csv('DOC_targets.csv')
taxa = pd.read_csv('selected_taxa.csv')['Taxa'].values

#%% run through permutations with NN model
r_values_NN = []
r_values_RF = []
r_values_NN_tuned = []
r_values_RF_tuned = []
r_values_NN_FS = []
r_values_RF_FS = []
permutation = 1

for train_perm, test_perm in zip(train_perms, test_perms):

    print('testing permutation {}'.format(permutation))
    permutation += 1
    # import training and test data as pandas dataframes
    train_data = pd.read_csv(train_path + train_perm)
    test_data = pd.read_csv(test_path + test_perm)

    # instantiate model class
    model = Model('Random Forest', train_data, test_data, targets)
    model.tune_hyper_params()

    # train model
    model.train_model()

    # test model
    Y_true, Y_pred_test, r = model.test_model(plot=False)
    r_values_RF.append(r)
    print(np.mean(r_values_RF))

    # instantiate model class
    model = Model('Neural Network', train_data, test_data, targets)
    model.tune_hyper_params()

    # train model
    model.train_model()

    # test model
    Y_true, Y_pred_test, r = model.test_model(plot=False)
    r_values_NN.append(r)
    print(np.mean(r_values_NN))
    '''
    # instantiate model class
    model = Model('Random Forest', train_data, test_data, targets)

    # set hyper parameters for model
    model.tune_hyper_params()

    # train model
    model.train_model()

    # test model
    Y_true, Y_pred_test, r = model.test_model(plot=False)
    r_values_RF_tuned.append(r)
    print(np.mean(r_values_RF_tuned))

    # instantiate model class
    model = Model('Neural Network', train_data, test_data, targets)

    # set hyper parameters for model
    print(model.tune_hyper_params())

    # train model
    model.train_model()

    # test model
    Y_true, Y_pred_test, r = model.test_model(plot=False)
    r_values_NN_tuned.append(r)
    print(np.mean(r_values_NN_tuned))

    # perform feature selection
    FS = FeatureSelection(train_data, test_data, targets)
    FS_results, names_IS, names_RF, names_NN  = FS.FeatureSelectionTable(return_names=True)
    N_keep = len(names_IS)
    full_set = list(names_RF[:N_keep]) + list(names_NN[:N_keep]) + list(names_IS[:N_keep])
    unique, counts = np.unique(full_set, return_counts=True)
    taxa = unique[counts==3]
    '''

    # reduce feature set
    train_data_FS = train_data[taxa]
    train_data_FS.insert(0, 'Sample ID', train_data[train_data.columns[0]])

    test_data_FS = test_data[taxa]
    test_data_FS.insert(0, 'Sample ID', test_data[test_data.columns[0]])

    # instantiate model class
    model = Model('Random Forest', train_data_FS, test_data_FS, targets)

    # set hyper parameters for model
    model.tune_hyper_params()

    # train model
    model.train_model()

    # test model
    Y_true, Y_pred_test, r = model.test_model(plot=False)
    r_values_RF_FS.append(r)
    print(np.mean(r_values_RF_FS))

    # instantiate model class
    model = Model('Neural Network', train_data_FS, test_data_FS, targets)

    # set hyper parameters for model
    print(model.tune_hyper_params())

    # train model
    model.train_model()

    # test model
    Y_true, Y_pred_test, r = model.test_model(plot=False)
    r_values_NN_FS.append(r)
    print(np.mean(r_values_NN_FS))
    t, p = ttest_ind(r_values_RF_FS, r_values_NN_FS)
    print(p)

#%% generate histogram of test results

plt.hist(r_values_RF)
plt.xlabel("Pearson's correlation coefficient")
plt.ylabel("frequency")
plt.xlim([0, 1])
plt.title('Random Forest')
plt.savefig('histograms/RF_DOC_prediction_hist.png', dpi=100)
np.savetxt('histograms/RF_r_values.csv', r_values_RF)
plt.close()
#plt.show()

#%% generate histogram of test results

plt.hist(r_values_NN)
plt.xlabel("Pearson's correlation coefficient")
plt.ylabel("frequency")
plt.xlim([0, 1])
plt.title('Neural Network')
plt.savefig('histograms/NN_DOC_prediction_hist.png', dpi=100)
np.savetxt('histograms/NN_r_values.csv', r_values_NN)
plt.close()
#plt.show()

#%% generate histogram of test results
'''
plt.hist(r_values_RF_tuned)
plt.xlabel("Pearson's correlation coefficient")
plt.ylabel("frequency")
plt.xlim([0, 1])
plt.title('Random Forest')
plt.savefig('histograms/RF_DOC_prediction_hist_tuned.png', dpi=100)
np.savetxt('histograms/RF_r_values_tuned.csv', r_values_RF_tuned)
plt.close()
#plt.show()
'''
#%% generate histogram of test results
'''
plt.hist(r_values_NN_tuned)
plt.xlabel("Pearson's correlation coefficient")
plt.ylabel("frequency")
plt.xlim([0, 1])
plt.title('Neural Network')
plt.savefig('histograms/NN_DOC_prediction_hist_tuned_decay.png', dpi=100)
np.savetxt('histograms/NN_r_values_tuned_decay.csv', r_values_NN_tuned)
plt.close()
#plt.show()
'''
#%% generate histogram of test results

plt.hist(r_values_RF_FS)
plt.xlabel("Pearson's correlation coefficient")
plt.ylabel("frequency")
plt.xlim([0, 1])
plt.title('Random Forest')
plt.savefig('histograms/RF_DOC_prediction_hist_FS.png', dpi=100)
np.savetxt('histograms/RF_r_values_FS.csv', r_values_RF_FS)
plt.close()
#plt.show()

#%% generate histogram of test results

plt.hist(r_values_NN_FS)
plt.xlabel("Pearson's correlation coefficient")
plt.ylabel("frequency")
plt.xlim([0, 1])
plt.title('Neural Network')
plt.savefig('histograms/NN_DOC_prediction_hist_FS.png', dpi=100)
np.savetxt('histograms/NN_r_values_FS.csv', r_values_NN_FS)
plt.close()
#plt.show()

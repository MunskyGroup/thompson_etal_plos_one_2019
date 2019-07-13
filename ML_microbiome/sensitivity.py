# import packages for uploading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
# import ML_microbiome package
import importlib
import ML_microbiome
importlib.reload(ML_microbiome)
from ML_microbiome import Model

# import training and test data as pandas dataframes
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
targets = pd.read_csv('DOC_targets.csv')

# create a df with results
results_df = pd.DataFrame()

taxa = pd.read_csv('selected_taxa.csv')['Taxa'].values

train_data_FS = train_data[taxa]
train_data_FS.insert(0, 'Sample ID', train_data[train_data.columns[0]])

test_data_FS = test_data[taxa]
test_data_FS.insert(0, 'Sample ID', test_data[test_data.columns[0]])

#%%

# instantiate model class
model = Model('Neural Network', train_data_FS, test_data_FS, targets)

Rvals_NN, Rstd_NN, cuts, NS = model.sensitivity(tune=False, plot=True)

#%% instantiate model class

model = Model('Random Forest', train_data, test_data, targets)

Rvals_RF, Rstd_RF, cuts, NS = model.sensitivity(tune=True)
N_train_samples = np.array(cuts*NS, np.int)

#%% plot NN results

plt.errorbar(N_train_samples, Rvals_NN, yerr=Rstd_NN, linestyle="None", marker='o')
plt.xlabel("Fraction of Training Set")
plt.ylabel("Performance with Test Data")
plt.title("Neural Network")
plt.ylim([0, 1])
plt.savefig('figures/NN_sensitivity.png', dpi=100)
plt.close()

#%% plot RF results

plt.errorbar(N_train_samples, Rvals_RF, yerr=Rstd_RF, linestyle="None", marker='o')
plt.xlabel("Fraction of Training Set")
plt.ylabel("Performance with Test Data")
plt.title("Random Forest")
plt.ylim([0, 1])
plt.savefig('figures/RF_sensitivity.png', dpi=100)
plt.close()

results_df['Rvals_NN'] = Rvals_NN
results_df['Rstd_NN'] = Rstd_NN
results_df['Rvals_RF'] = Rvals_RF
results_df['Rstd_RF'] = Rstd_RF
results_df['Fraction'] = cuts

results_df.to_csv('figures/sensitivity_tuned.csv', index=False)

import numpy as np
from scipy.stats import ttest_ind as ttest

nn = np.genfromtxt('NN_r_values.csv')
rf = np.genfromtxt('RF_r_values.csv')
nnfs = np.genfromtxt('NN_r_values_FS.csv')
rffs = np.genfromtxt('RF_r_values_FS.csv')

#%%

print('NN r mean: {:.3f}, std: {:.3f}'.format(np.mean(nn), np.std(nn)))
print('RF r mean: {:.3f}, std: {:.3f}'.format(np.mean(rf), np.std(rf)))
print('NN FS r mean: {:.3f}, std: {:.3f}'.format(np.mean(nnfs), np.std(nnfs)))
print('RF FS r mean: {:.3f}, std: {:.3f}'.format(np.mean(rffs), np.std(rffs)))

#%%

ttest(rf, rffs)
ttest(nn, nnfs)

#%% determine percentage of samples with greater prediction accuracies

P_improved_nn = sum(nnfs > nn) / len(nn)
P_improved_rf = sum(rffs > rf) / len(rf) 

print(P_improved_nn)
print(P_improved_rf)

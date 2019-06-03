import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sensitivity_data = pd.read_csv('sensitivity_tuned.csv')

cuts = sensitivity_data['Fraction'].values
NS = 257
N_samples = np.array(NS*cuts, np.int)

#%% plot RF figure
Rvals_RF = sensitivity_data['Rvals_RF'].values
Rstd_RF = sensitivity_data['Rstd_RF'].values
plt.errorbar(N_samples, Rvals_RF, capsize=3, markersize=4, alpha=0.85,
    linestyle='none', marker='D', yerr = Rstd_RF)
plt.xlabel('Number of Samples in Training Set')
plt.ylabel('Prediction Performance')
plt.ylim([0, 1])
plt.title('Random Forest')
plt.savefig('RF_sensitivity.png', dpi=100)
plt.show()

#%% plot NN figure
Rvals_NN = sensitivity_data['Rvals_NN'].values
Rstd_NN = sensitivity_data['Rstd_NN'].values
plt.errorbar(N_samples, Rvals_NN, capsize=3, markersize=4, alpha=0.85,
    linestyle='none', marker='D', yerr = Rstd_NN)
plt.xlabel('Number of Samples in Training Set')
plt.ylabel('Prediction Performance')
plt.ylim([0, 1])
plt.title('Neural Network')
plt.savefig('NN_sensitivity.png', dpi=100)
plt.show()

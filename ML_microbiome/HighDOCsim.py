# import packages for uploading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import ML_microbiome package
#import importlib
import ML_microbiome
#importlib.reload(ML_microbiome)
from ML_microbiome import Model
from scipy import stats

np.random.seed(123)

# import training and test data as pandas dataframes
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
targets = pd.read_csv('DOC_targets.csv')

taxa = pd.read_csv('selected_taxa.csv')['Taxa'].values

train_data_FS = train_data[taxa]
train_data_FS.insert(0, 'Sample ID', train_data[train_data.columns[0]])

test_data_FS = test_data[taxa]
test_data_FS.insert(0, 'Sample ID', test_data[test_data.columns[0]])

# instantiate model class
model = Model('Neural Network', train_data_FS, test_data_FS, targets)
model.set_model_params({'p_dropout':0})

#
# determine feature importance vector
train_data_FS, test_data_FS, names, NN_importances = model.feature_selection()

# tune hyper parameters for model, train, and test
results = model.tune_hyper_params()
print(results)
Y_train, Y_pred_train, R_train = model.train_model()
Y_test, Y_pred_test, R = model.test_model(plot=False)

#%% vary x_orig to increase Y based on gradient
Nsim = 25
alpha_array = np.linspace(0, 5, Nsim)
Y_sim = np.zeros(Nsim)

# only consider increasing populations
dX = (NN_importances - np.mean(NN_importances)) / np.std(NN_importances)

# generate random vectors for use with random perturbations
v = []
Nrandperturbation = 3
for i in range(Nrandperturbation):
    v.append(np.random.randn(len(dX)))

X_orig = train_data_FS.values[np.argsort(Y_train)[-1], 1:]
for i, alpha in enumerate(alpha_array):
    X_new = X_orig + alpha*dX
    Y_sim[i] = model.predict(X_new[np.newaxis, :])
plt.plot(alpha_array, Y_sim, label='Simulated community (a)')

for j in range(Nrandperturbation):
    X_community_C = []
    for i, alpha in enumerate(alpha_array):
        X_new = X_orig + alpha*v[j]
        X_community_C.append(X_new)
        Y_sim[i] = model.predict(X_new[np.newaxis, :])
    plt.plot(alpha_array, Y_sim, 'C0--', alpha=0.5)

X_orig = train_data_FS.values[np.argsort(Y_train)[0], 1:]
for i, alpha in enumerate(alpha_array):
    X_new = X_orig + alpha*dX
    Y_sim[i] = model.predict(X_new[np.newaxis, :])
plt.plot(alpha_array, Y_sim, label='Simulated community (b)')

for j in range(Nrandperturbation):
    for i, alpha in enumerate(alpha_array):
        X_new = X_orig + alpha*v[j]
        X_community_C.append(X_new)
        Y_sim[i] = model.predict(X_new[np.newaxis, :])
    plt.plot(alpha_array, Y_sim, 'C1--', alpha=0.5)

plt.plot([], [], 'k--', alpha=0.5, label='Random perturbation')

plt.xlabel(r"$\alpha$")
plt.ylabel("Predicted DOC")
plt.title("Neural Network")
plt.legend(loc=4)
#plt.savefig('NN_doc_sim.png', dpi=100)
plt.show()
#plt.close()

#%% cross validate that X_new represents a high DOC sample with RF model

# instantiate model class
model_RF = Model('Random Forest', train_data_FS, test_data_FS, targets)
Y_train, Y_pred_train, R_train = model_RF.train_model()
Y_test, Y_pred_test, R = model_RF.test_model(plot=False)

X_orig = train_data_FS.values[np.argsort(Y_train)[-1], 1:]
for i, alpha in enumerate(alpha_array):
    X_new = X_orig + alpha*dX
    Y_sim[i] = model_RF.predict(X_new[np.newaxis, :])
plt.plot(alpha_array, Y_sim, label='Simulated community (a)')

for j in range(Nrandperturbation):
    for i, alpha in enumerate(alpha_array):
        X_new = X_orig + alpha*v[j]
        Y_sim[i] = model_RF.predict(X_new[np.newaxis, :])
    plt.plot(alpha_array, Y_sim, 'C0--', alpha=0.5)

X_orig = train_data_FS.values[np.argsort(Y_train)[0], 1:]
for i, alpha in enumerate(alpha_array):
    X_new = X_orig + alpha*dX
    Y_sim[i] = model_RF.predict(X_new[np.newaxis, :])
plt.plot(alpha_array, Y_sim, label='Simulated community (b)')

for j in range(Nrandperturbation):
    for i, alpha in enumerate(alpha_array):
        X_new = X_orig + alpha*v[j]
        Y_sim[i] = model_RF.predict(X_new[np.newaxis, :])
    plt.plot(alpha_array, Y_sim, 'C1--', alpha=0.5)

plt.plot([], [], 'k--', alpha=0.5, label='Random perturbation')

plt.xlabel(r"$\alpha$")
plt.ylabel("Predicted DOC")
plt.title("Random Forest")
plt.legend(loc=4)
#plt.savefig('RF_doc_sim.png', dpi=100)
plt.show()
#plt.close()

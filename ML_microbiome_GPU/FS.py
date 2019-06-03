import pandas as pd
import numpy as np
from scipy.stats import linregress, norm
import matplotlib.pyplot as plt
# import ML_microbiome package
import importlib
import ML_microbiome
#importlib.reload(ML_microbiome)
from ML_microbiome import Model, IndicatorSpecies, FeatureSelection
# import plotting libraries
import matplotlib.pyplot as plt
import matplotlib_venn
from matplotlib_venn import venn3
plt.rcParams.update({'font.size': 11})

# import training and test data as pandas dataframes
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
targets = pd.read_csv('DOC_targets.csv')

all = pd.concat([train_data, test_data])
N_features_total = sum(sum(all.values[:,1:], 0) > 0)
N_features_train = sum(sum(train_data.values[:,1:], 0) > 0)

#%%

FS = FeatureSelection(train_data, test_data, targets)
FS_results, names_IS, names_RF, names_NN  = FS.FeatureSelectionTable(return_names=True)
print(FS_results.head(10).to_latex())

FS_results.sort_values(by='IS stat', ascending=False).to_csv('FS_table.csv', index=False)

#%% use FS_results to create FS venn diagram and FS plot figures

N_keep = len(names_IS)
print(N_keep)

venn3([set(names_NN[:N_keep]), set(names_RF[:N_keep]), set(names_IS[:N_keep])],
    ('Neural Network', 'Random Forest', 'Indicator Species'))
#plt.savefig('figures/FS/FS_venn.png', dpi=100)
plt.show()

#%% generate figure to illustrate degree of overlap as feature rank decreases

N_keep = len(names_IS)
#N_keep = 10

NN_shared = []
RF_shared = []
NN_RF_shared = []
NN_RF_IS_shared = []

for i in range(1, N_keep):
    full_set = list(names_NN[:i]) + list(names_IS[:i])
    unique, counts = np.unique(full_set, return_counts=True)
    NN_shared.append(sum(counts==2))

for i in range(1, N_keep):
    full_set = list(names_RF[:i]) + list(names_IS[:i])
    unique, counts = np.unique(full_set, return_counts=True)
    RF_shared.append(sum(counts==2))

for i in range(1, N_keep):
    full_set = list(names_RF[:i]) + list(names_NN[:i])
    unique, counts = np.unique(full_set, return_counts=True)
    NN_RF_shared.append(sum(counts==2))

for i in range(1, N_keep):
    full_set = list(names_RF[:i]) + list(names_NN[:i]) + list(names_IS[:i])
    unique, counts = np.unique(full_set, return_counts=True)
    NN_RF_IS_shared.append(sum(counts==3))

def E_shared(N):
    return (np.arange(N)/N)**2 * np.arange(N)

def ss3(n, N, iters):

    n_shared = np.zeros([iters, n])

    for iter in range(iters):
        # define 3 random lists of length N
        l1 = np.random.permutation(N)
        l2 = np.random.permutation(N)
        l3 = np.random.permutation(N)

        # randomly sample range(n) values from the 3 lists
        for i in range(n):
            l = list(l1[:i]) + list(l2[:i]) + list(l3[:i])
            unique, counts = np.unique(l, return_counts=True)
            n_shared[iter, i] = sum(counts==3)
    return n_shared

def ss2(n, N, iters):

    n_shared = np.zeros([iters, n])

    for iter in range(iters):
        # define 2 random lists of length N
        l1 = np.random.permutation(N)
        l2 = np.random.permutation(N)

        # randomly sample range(n) values from the 3 lists
        for i in range(n):
            l = list(l1[:i]) + list(l2[:i])
            unique, counts = np.unique(l, return_counts=True)
            n_shared[iter, i] = sum(counts==2)
    return n_shared

nsim = 1000
sim2 = ss2(N_keep, N_features_total, nsim)
sim3 = ss3(N_keep, N_features_total, nsim)
#%%
confidence_level = 0.995 # (for the 99 % confidence interval)
zval = norm.ppf(confidence_level)
CI_2 = zval * (np.std(sim2, 0) / np.sqrt(nsim))
CI_3 = zval * (np.std(sim3, 0) / np.sqrt(nsim))

# plot experimental results
nn_shared, = plt.plot(range(1, N_keep), NN_shared)
rf_shared, = plt.plot(range(1, N_keep), RF_shared)
nn_rf_shared, = plt.plot(range(1, N_keep), NN_RF_shared)
nn_rf_is_shared, = plt.plot(range(1, N_keep), NN_RF_IS_shared)

# plot simulation results
mc2, = plt.plot(range(N_keep), np.mean(sim2, 0), 'k', linewidth=0.5)
ci2, = plt.plot([], [], color='C4', linewidth=5)
plt.fill_between(range(N_keep), np.mean(sim2, 0)-CI_2, np.mean(sim2, 0)+CI_2, color='C4')

mc3, = plt.plot(range(N_keep), np.mean(sim3, 0), 'k', linewidth=0.5)
ci3, = plt.plot([], [], color='C5', linewidth=5)
plt.fill_between(range(N_keep), np.mean(sim3, 0)-CI_3, np.mean(sim3, 0)+CI_3, color='C5')

plt.plot(range(1, N_keep), range(1, N_keep), 'k--', label='Perfect Agreement')
plt.xlim([0, N_keep])
plt.ylim([0, N_keep])
plt.xlabel('Number of Features')
plt.ylabel('Number of Shared Features')

plt.legend([nn_shared, rf_shared, nn_rf_shared, nn_rf_is_shared, (ci2, mc2), (ci3, mc3)],
    ["NN and IS", "RF and IS", "RF and NN", "RF, NN, and IS", "Expected by chance (2 sets)", "Expected by chance (3 sets)"])

#plt.savefig('figures/FS/FeatureSelectionPlot_mc_99ci.png', dpi=100)
#plt.close()
plt.show()
#%% generate and save just the Monte Carlo simulation figure

# plot simulation results
mc2, = plt.plot(range(N_keep), np.mean(sim2, 0), 'k', linewidth=0.5)
ci2, = plt.plot([], [], color='C4', linewidth=5)
plt.fill_between(range(N_keep), np.mean(sim2, 0)-CI_2, np.mean(sim2, 0)+CI_2, color='C4')

mc3, = plt.plot(range(N_keep), np.mean(sim3, 0), 'k', linewidth=0.5)
ci3, = plt.plot([], [], color='C5', linewidth=5)
plt.fill_between(range(N_keep), np.mean(sim3, 0)-CI_3, np.mean(sim3, 0)+CI_3, color='C5')

plt.xlabel('Number of Features')
plt.ylabel('Number of Shared Features')

plt.legend([(ci2, mc2), (ci3, mc3)],
    ["Expected by chance (2 sets)", "Expected by chance (3 sets)"])

plt.savefig("figures/FS/montecarlosim.png", dpi=300)
plt.show()

#%% lets find out how well NN could correlate features with target variable

NN_site_labels = []
for i, importance in enumerate(FS_results["NN Importance"]):
    if importance < 0:
        NN_site_labels.append("Low")
    else:
        NN_site_labels.append("High")

correct = 0
for NN_label, IS_label in zip(NN_site_labels, FS_results['IS Site Label'].values):
    if NN_label == IS_label:
        correct += 1

Pcorrect = correct / len(NN_site_labels)
print(Pcorrect)

#%% generate histogram of abundance of all features and selected features
OTUtable = pd.concat([train_data, test_data], axis=0)

# remove features with no abundance counts
non_zeros = np.sum(OTUtable.values[:,1:], 0) > 0
feature_table = OTUtable.values[:,1:][:, non_zeros]
avg_abundance = np.array(np.mean(feature_table , 0), np.float)

# calculate average abundance of each OTU among all samples
joint_OTUs = unique[counts==3]
reduced_feature_table = OTUtable[joint_OTUs].values
avg_abundance_select = np.array(np.mean(reduced_feature_table, 0), np.float)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

edges,bins = np.histogram(np.log10(avg_abundance),10)
ax1.hist(np.log10(avg_abundance), bins=bins, alpha=.5, label='All Taxa')

edges_reduced, bins = np.histogram(np.log10(avg_abundance_select), bins)
ax1.hist(np.log10(avg_abundance_select), bins=bins, alpha=.5, label='Selected Taxa')

ax1.set_ylabel('Number of Features')
ax1.set_xlabel(r'$log_{10}$ (abundance)')
ax1.set_yscale('log')
ax1.legend(loc=2)

# include ratio
def center(bins):
    centers = []
    for i in range(1, len(bins)):
        centers.append(np.mean([bins[i-1], bins[i]]))
    return centers

ax2.scatter(center(bins), edges_reduced/edges, s=25, facecolors='none', edgecolors='k', marker='o', label='Fraction selected')
ax2.plot(center(bins), edges_reduced/edges, 'k', alpha=0.5)
ax2.set_ylabel('Fraction selected')
ax2.legend(loc=1)
ax2.set_ylim([0, 1])

plt.savefig('figures/FS/abundance_histogram.png', dpi=110)
#plt.close()
plt.show()

#df = pd.DataFrame(joint_OTUs, columns=['Taxa'])
#df.to_csv('selected_taxa.csv', index=False)
#%%

# plot % presence / absence histogram
K, NF_all = feature_table.shape
P_present = sum(feature_table > 0, 0) / K
K, NF_reduced = reduced_feature_table.shape
P_present_reduced = sum(reduced_feature_table > 0, 0) / K

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

edges,bins = np.histogram(P_present,10)
ax1.hist(P_present, bins=bins, alpha=.5, label='All Taxa')

edges_reduced, bins = np.histogram(P_present_reduced, bins)
ax1.hist(P_present_reduced, bins=bins, alpha=.5, label='Selected Taxa')

ax1.set_ylabel('Number of Features')
ax1.set_xlabel("Prevalence")
ax1.set_yscale('log')
ax1.legend(loc=2)

# include ratio
def center(bins):
    centers = []
    for i in range(1, len(bins)):
        centers.append(np.mean([bins[i-1], bins[i]]))
    return centers

ax2.scatter(center(bins), edges_reduced/edges, s=25, facecolors='none', edgecolors='k', marker='o', label='Fraction selected')
ax2.plot(center(bins), edges_reduced/edges, 'k', alpha=0.5)
ax2.set_ylabel('Fraction selected')
ax2.legend(loc=1)
ax2.set_ylim([0, 1])

plt.savefig('figures/FS/prevalence_histogram.png', dpi=110)
#plt.close()
plt.show()

#%%

# define function to bin sets of n samples and return array with avg, std dev
def bin_samples(samples, n=5):
    avgs = []
    stds = []
    set = []
    for i, sample in zip(range(1, len(samples)+1), samples):
        set.append(sample)
        if i%n == 0:
            avgs.append(np.mean(set))
            stds.append(np.std(set))
            set = []
    return np.array(avgs), np.array(stds)

# define function to get r values based on the list of OTUs
def get_r_vals(shared_OTUs):
    # create model and make predictions with reduced feature set
    min_features = 10
    r_vals_RF = []
    r_vals_NN = []
    N_features = np.arange(min_features, len(shared_OTUs))

    for NF in N_features:
        NF = int(np.ceil(NF))
        train_data_FS = train_data[shared_OTUs[:NF+1]]
        train_data_FS.insert(0, 'Sample ID', train_data[train_data.columns[0]])

        test_data_FS = test_data[shared_OTUs[:NF+1]]
        test_data_FS.insert(0, 'Sample ID', test_data[test_data.columns[0]])

        # instantiate model class
        model = Model('Neural Network', train_data_FS, test_data_FS, targets)
        model.tune_hyper_params()
        # train model
        model.train_model()
        # test model
        y_test, y_pred_test, r = model.test_model(plot=False)
        # save r values over range of number of selected features
        r_vals_NN.append(r)

        # instantiate model class
        model = Model('Random Forest', train_data_FS, test_data_FS, targets)
        model.tune_hyper_params()
        # train model
        model.train_model()
        # test model
        y_test, y_pred_test, r = model.test_model(plot=False)
        # save r values over range of number of selected features
        r_vals_RF.append(r)

    return r_vals_NN, r_vals_RF, N_features

#%%
'''
shared_OTUs_blue = []
for i, name_IS in enumerate(names_IS):
    if name_IS in names_NN[:len(names_IS)]:
        shared_OTUs_blue.append(name_IS)

r_vals_NN, r_vals_RF, N_features = get_r_vals(shared_OTUs_blue)

N_ftrs_axis, N_ftrs_std = bin_samples(N_features)
mean_r_vals_NN, std_r_vals_NN = bin_samples(r_vals_NN)
mean_r_vals_RF, std_r_vals_RF = bin_samples(r_vals_RF)
#%%
plt.figure(figsize=(10, 4))
plt.errorbar(N_ftrs_axis, mean_r_vals_RF, linestyle='none', marker='o', yerr = std_r_vals_RF, label="Random Forest")
plt.errorbar(N_ftrs_axis, mean_r_vals_NN, linestyle='none', marker='o', yerr = std_r_vals_NN, label="Neural Network")
#plt.axvline(x=len(joint_OTUs)
plt.xlabel('Number of Features')
plt.ylabel('Prediction Performance')
plt.legend(loc=4)
plt.savefig('figures/FS/PerformanceVsFeatures_Blue.png', dpi=100)
plt.show()
'''
#%%
'''
shared_OTUs_Orange = []
for i, name_IS in enumerate(names_IS):
    if name_IS in names_RF[:len(names_IS)]:
        shared_OTUs_Orange.append(name_IS)

r_vals_NN, r_vals_RF, N_features = get_r_vals(shared_OTUs_Orange)

N_ftrs_axis, N_ftrs_std = bin_samples(N_features)
mean_r_vals_NN, std_r_vals_NN = bin_samples(r_vals_NN)
mean_r_vals_RF, std_r_vals_RF = bin_samples(r_vals_RF)

plt.figure(figsize=(10, 4))
plt.errorbar(N_ftrs_axis, mean_r_vals_RF, linestyle='none', marker='o', yerr = std_r_vals_RF, label="Random Forest")
plt.errorbar(N_ftrs_axis, mean_r_vals_NN, linestyle='none', marker='o', yerr = std_r_vals_NN, label="Neural Network")
#plt.axvline(x=len(joint_OTUs)
plt.xlabel('Number of Features')
plt.ylabel('Prediction Performance')
plt.legend(loc=4)
plt.savefig('figures/FS/PerformanceVsFeatures_Orange.png', dpi=100)
plt.show()
'''
#%%
'''
shared_OTUs_red = []
for i, name_IS in enumerate(names_IS):
    if name_IS in names_NN[:len(names_IS)] and name_IS in names_RF[:len(names_IS)]:
        shared_OTUs_red.append(name_IS)

r_vals_NN, r_vals_RF, N_features = get_r_vals(shared_OTUs_red)

#%%
N_ftrs_axis, N_ftrs_std = bin_samples(N_features)
mean_r_vals_NN, std_r_vals_NN = bin_samples(r_vals_NN)
mean_r_vals_RF, std_r_vals_RF = bin_samples(r_vals_RF)

#plt.figure(figsize=(10, 4))
plt.errorbar(N_ftrs_axis, mean_r_vals_RF, capsize=3, markersize=4, alpha=0.75, linestyle='none', marker='D', yerr = std_r_vals_RF, label="Random Forest")
plt.errorbar(N_ftrs_axis, mean_r_vals_NN, capsize=3, markersize=4, alpha=0.75, linestyle='none', marker='D', yerr = std_r_vals_NN, label="Neural Network")
#plt.axvline(x=len(joint_OTUs)
plt.xlabel('Number of Features')
plt.ylabel('Prediction Performance')
plt.legend(loc=4)
plt.savefig('figures/FS/PerformanceVsFeatures_Red.png', dpi=100)
plt.show()

#%% save results to dataframe
df = pd.DataFrame()
df['N features'] = N_ftrs_axis
df['NN r values mean'] = mean_r_vals_NN
df['NN r values std'] = std_r_vals_NN
df['RF r values mean'] = mean_r_vals_RF
df['RF r values std'] = std_r_vals_RF

df.to_csv('figures/FS/PerformanceVsFeatures_data.csv', index=False)
'''
#%%
'''
train_data_FS = train_data[joint_OTUs]
train_data_FS.insert(0, 'Sample ID', train_data[train_data.columns[0]])

test_data_FS = test_data[joint_OTUs]
test_data_FS.insert(0, 'Sample ID', test_data[test_data.columns[0]])

# instantiate model class
model = Model('Neural Network', train_data_FS, test_data_FS, targets)

# train model
ytr, yptr, r = model.train_model()

# test model
y, yp, r = model.test_model(plot=True)

lr = linregress(ytr, yptr)
yp *= lr[0]

plt.scatter(ytr, yptr)
plt.scatter(y, yp)
'''

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# import statistics packages
from scipy import stats
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import norm as sp_norm

# import sklearn packages
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# import neural network class
from .network import Network

# import Bayesian linear regression
from .LR import BLR

np.random.seed(123)

class Model:

    def __init__(self, model, train_data, test_data, targets, test_targets=None, standardize=True, whiten=False, richness=False):

        # remove unlabeled samples from training and testing data
        labeled_samples = targets.columns.values[1:]
        train_samples = train_data.values[:, 0]
        test_samples = test_data.values[:, 0]

        inds = np.in1d(train_samples, labeled_samples)
        n_remove = len(train_samples) - np.sum(inds)
        if n_remove > 0:
            print("Removing {} unlabeled training samples".format(n_remove))
            train_data = train_data.iloc[inds, :]

        inds = np.in1d(test_samples, labeled_samples)
        n_remove = len(test_samples) - np.sum(inds)
        if n_remove > 0:
            print("Removing {} unlabeled testing samples".format(n_remove))
            test_data = test_data.iloc[inds, :]

        if richness: # include species abundance
            richness = np.sum(train_data.values[:, 1:] > 0, 1)
            train_data['richness'] = richness

            richness = np.sum(test_data.values[:, 1:] > 0, 1)
            test_data['richness'] = richness

        if test_targets is None:
            test_targets = targets

        self.model_name = model
        self.train_data = train_data
        self.test_data = test_data

        # store sample IDs for future reference
        self.train_samples = train_data.values[:, 0]
        self.test_samples = test_data.values[:, 0]

        # get feature names
        feature_names = train_data.columns.values[1:]

        # organize training and test data with associated target variables
        X_train = np.array(train_data.values[:, 1:], np.float)
        X_test = np.array(test_data.values[:, 1:], np.float)

        train_sample_IDs = train_data.values[:, 0]
        Y_train = np.ravel(targets[train_sample_IDs].values)

        test_sample_IDs = test_data.values[:, 0]
        Y_test = np.ravel(test_targets[test_sample_IDs].values)

        # remove uninformative features and standardize X and Y
        informative = np.std(X_train, 0)>0
        X_train = X_train[:, informative]
        X_test = X_test[:, informative]

        # remove uninformative features from train and test DataFrames
        train_df = pd.DataFrame(data=X_train, columns=feature_names[informative])
        train_df.insert(0, 'sample ID', self.train_samples)
        self.train_data = train_df
        test_df = pd.DataFrame(data=X_test, columns=feature_names[informative])
        test_df.insert(0, 'sample ID', self.test_samples)
        self.test_data = test_df

        if standardize:
            # standardize X and Y relative to training data
            self.MNX = np.mean(X_train, 0)
            self.STX = np.std(X_train, 0)
            MNY = np.mean(Y_train)
            STY = np.std(Y_train)

            self.MNX_test = np.mean(X_test, 0)
            self.STX_test = np.std(X_test, 0)
            MNY_test = np.mean(Y_test)
            STY_test = np.std(Y_test)

            self.X_train = (X_train - self.MNX) / self.STX
            self.X_test = (X_test - self.MNX) / self.STX
            self.Y_train = (Y_train - MNY) / STY
            self.Y_test = (Y_test - MNY) / STY
        elif whiten:
            pca = PCA(whiten=True)
            pca.fit(X_train)
            self.X_train = pca.transform(X_train)
            self.X_test = pca.transform(X_test)
            MNY = np.mean(Y_train)
            STY = np.std(Y_train)
            self.Y_train = (Y_train - MNY) / STY
            self.Y_test = (Y_test - MNY) / STY
        else:
            self.X_train = X_train
            self.X_test = X_test
            self.Y_train = Y_train
            self.Y_test = Y_test

        NS_train, NF = self.X_train.shape

        # model is a string specifying neural network or random forest
        if model == 'Random Forest':
            self.model = None
            self.model = RandomForest(self)
        elif model == 'Neural Network':
            self.model = None
            self.model = NeuralNetwork(self, NF)
        elif model == 'Bayesian Linear Regression':
            self.model = None
            self.model = BayesianLinearRegression(self, self.X_train, self.Y_train)

    def tune_hyper_params(self):
        # tune hyper parameters using Scikit Learn's random search algorithm
        # print("Tuning hyper-parameters...")
        return self.model.tune(self.X_train, self.Y_train)

    def train_model(self):
        # use training data to train the model
        return self.model.train(self.X_train, self.Y_train)

    def test_model(self, plot=True):
        # use testing data to test the model
        return self.model.test(self.X_train, self.X_test, self.Y_train, self.Y_test, plot=plot)

    def predict(self, X):
        # use the model to make predictions on data set without labels
        return self.model.predict(X)

    def feature_selection(self, keep='all', iterations=50):
        # return feature selection results from chosen model
        X_train_select, selected_features, importances = self.model.select_features(self.X_train, self.Y_train, keep, iterations=iterations)

        # update model with selected features
        self.X_train = X_train_select
        self.X_test = self.X_test[:, selected_features]

        # determine names of kept features
        names = self.train_data.columns.values[1:][selected_features]

        # create pandas dataframes of feature reduced data sets
        X_train_df = pd.DataFrame(data=self.X_train, columns=names)
        X_train_df.insert(0, 'sample ID', self.train_samples)

        X_test_df = pd.DataFrame(data=self.X_test, columns=names)
        X_test_df.insert(0, 'sample ID', self.test_samples)

        return X_train_df, X_test_df, names, importances

    def sensitivity(self, N_cuts=10, N_tests=10, tune=False, plot=False):
        # N_cuts defines the number of times to split the training data set
        # N_tests defines the number of times to test each training set split

        NS, NF = self.X_train.shape
        # define number of splits of training set
        cuts = np.linspace(.1, 1, N_cuts)

        # create an empty array of R values and std_deviations
        R_vals = np.zeros(N_cuts)
        R_std = np.zeros(N_cuts)

        X_train_orig = self.X_train
        Y_train_orig = self.Y_train

        for i, cut in enumerate(cuts):
            R = np.zeros(N_tests)
            print("Training with {:.2f}% of data set".format(cut*100))
            for j in range(N_tests):
                print("Running Test " + str(j+1))
                randinds = np.random.permutation(NS)
                self.X_train = X_train_orig[randinds, :][:int(cut*NS),:]
                self.Y_train = Y_train_orig[randinds][:int(cut*NS)]
                self.model.reset()

                if tune:
                    self.tune_hyper_params()
                self.train_model()
                y_test, y_pred, R[j] = self.test_model(plot=False)

            R_vals[i] = np.mean(R)
            R_std[i] = np.std(R)

        # reset X_train and Y_train to original permutations
        self.X_train = X_train_orig
        self.Y_train = Y_train_orig

        if plot:
            plt.errorbar(cuts, R_vals, yerr=R_std, linestyle="None")
            plt.xlabel("Fraction of Training Set")
            plt.ylabel("Performance with Test Data")
            plt.title(self.model_name)
            plt.show()

        return R_vals, R_std, cuts, NS

    def plot(self, Y_train, Y_pred_train, Y_test, Y_pred_test, plot=True, Yerr=None):
        # plot linear fit of training data
        if plot:
            plt.subplot(1,2,1)
            plt.scatter(Y_train, Y_pred_train, facecolors='none', edgecolors='b', label='Data')
            #xlim = plt.xlim()
            #ylim = plt.ylim()
            pad = .25
            xlim = [min(Y_train)-pad, max(Y_train)+pad]
            ylim = [min(Y_train)-pad, max(Y_train)+pad]
            slope, intercept, r_value, p_value, std_err = stats.linregress(Y_train, Y_pred_train)
            x = np.linspace(min(xlim), max(xlim), 100)
            line = np.multiply(slope,x) + np.array(intercept)
            plt.plot(x, line, 'r', label='Fit')
            title_string = 'Training: R = %.3f' % r_value
            ylabel = 'Output = {0:.2f}*Target + {1:.2f}'.format(slope, intercept)
            plt.title(title_string)
            plt.xlabel('Target')
            plt.ylabel(ylabel)
            plt.xlim(xlim)
            plt.ylim(ylim)
            #plt.ylim([ymin-pad, ymax+pad])
            plt.legend()

            # plot test data results
            plt.subplot(1,2,2)
            slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test, Y_pred_test)
            x = np.linspace(min(xlim), max(xlim), 100)
            line = np.multiply(slope, x) + np.array(intercept)
            if Yerr is None:
                plt.scatter(Y_test, Y_pred_test, facecolors='none', edgecolors='b', label='Data')
            else:
                plt.errorbar(Y_test, Y_pred_test, linestyle='none', marker='o', yerr = Yerr, label='Data')
            plt.plot(x, line, 'r', label='Fit')
            title_string = 'Test: R = %.3f' % r_value
            ylabel = 'Output = {0:.2f}*Target + {1:.2f}'.format(slope, intercept)
            plt.title(title_string)
            plt.xlabel('Target')
            plt.ylabel(ylabel)
            plt.xlim(xlim)
            plt.ylim(ylim)
            #plt.ylim([ymin-pad, ymax+pad])
            plt.legend()
            # save or show figures
            plt.suptitle(self.model_name)
            plt.tight_layout(rect=[0, .03, 1, .95])
            plt.show()
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test, Y_pred_test)
        return r_value, p_value, std_err

    def get_params(self, deep=True):
        return {"X_train": self.X_train,
                "Y_train": self.Y_train,
                "X_test": self.X_test,
                "Y_test": self.Y_test}

    def set_model_params(self, params):
        self.model.set_model_params(params)

class RandomForest:

    def __init__(self, Model_Class):
        # load train and test features as numpy arrays
        self.RF = RandomForestRegressor(n_estimators=1000, n_jobs=4)
        self.Model_Class = Model_Class

    def reset(self):
        try:
            self.RF = RandomForestRegressor(
                n_estimators=1000,
                min_samples_leaf=self.params['min_samples_leaf'],
                min_samples_split=self.params['min_samples_split'],
                n_jobs=4)
        except:
            self.RF = RandomForestRegressor(n_estimators=1000, n_jobs=4)

    def train(self, X_train, Y_train):
        self.RF.fit(X_train, Y_train)
        Y_pred_train = self.RF.predict(X_train)
        m_train, intercept, r_value, p_value, std_err = stats.linregress(Y_train, Y_pred_train)

        return Y_train, Y_pred_train, r_value

    def test(self, X_train, X_test, Y_train, Y_test, plot=True):
        Y_pred_train = self.RF.predict(X_train)
        m_train, intercept, r_value, p_value, std_err = stats.linregress(Y_train, Y_pred_train)
        Y_pred_test = self.RF.predict(X_test) #/ m_train

        r_value, p_value, std_err = self.Model_Class.plot(Y_train, Y_pred_train, Y_test, Y_pred_test, plot=plot)
        return Y_test, Y_pred_test, r_value

    def predict(self, X):
        # make predictions on unlabeled data
        return self.RF.predict(X)

    def tune(self, X_train, Y_train):
        # define grid of possible hyper parameter values
        param_dist = {"min_samples_split": sp_randint(2, 11),
                      "min_samples_leaf": sp_randint(1, 11)}

        random_search = RandomizedSearchCV(self.RF, n_jobs=8,
            param_distributions = param_dist, cv=5)
        random_search.fit(X_train, Y_train)
        self.params = random_search.best_params_
        # update RF to match best hyper parameters
        self.RF = RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=self.params['min_samples_leaf'],
            min_samples_split=self.params['min_samples_split'],
            n_jobs=4)

        return self.params

    def select_features(self, X_train, Y_train, keep='all', iterations=50):
        # perform feature ranking with subsets of training data

        # define percentage of training data to keep when bootstrapping
        p_bootstrap = .8
        NS, NF = X_train.shape

        if keep == 'all':
            keep = NF
        elif keep > NF:
            keep = NF

        def get_importances():
            # keep p_bootstrap % of training sample set after shuffling
            rand_inds = np.random.permutation(NS)
            X_train_sample = X_train[rand_inds, :][:int(p_bootstrap*NS), :]
            Y_train_sample = Y_train[rand_inds][:int(p_bootstrap*NS)]
            # fit RF model to bootstrap sample of training data
            self.RF.fit(X_train_sample, Y_train_sample)
            importances = self.RF.feature_importances_
            self.reset()
            return importances
        feature_importances = Parallel(n_jobs=4)(delayed(get_importances)() for _ in range(iterations))

        # record mean feature importances
        mean_feature_importances = np.mean(feature_importances, 0)

        # sort importances and keep subset
        sorted_feature_importances = np.sort(mean_feature_importances)[::-1][:keep]

        # return X_train with kept features
        selected_features = np.argsort(np.abs(mean_feature_importances))[::-1][:keep]
        X_train_select = X_train[:, selected_features]

        return X_train_select, selected_features, sorted_feature_importances

    def set_model_params(self, params):
        self.RF.set_params(**params)

class NeuralNetwork:

    def __init__(self, Model_Class, NF):
        # load train and test features as numpy arrays
        self.Model_Class = Model_Class
        self.NF = NF
        self.net = Network(NF)

    def reset(self):
        try:
            self.net = Network(self.NF,
                nodes=self.params['nodes'],
                eta=self.params['eta'],
                lmbda=self.params['lmbda'],
                patience=self.params['patience'])
        except:
            self.net = Network(self.NF)

    def train(self, X_train, Y_train):
        # SGD inputs training_data,epochs,batch size,learning_rate
        self.net.set_params(verbose=False)
        self.net.fit(X_train, Y_train)
        Y_pred_train = self.net.predict(X_train)
        slope, intercept, r_value, p_value, std_err = stats.linregress(Y_train, Y_pred_train)

        return Y_train, Y_pred_train, r_value

    def test(self, X_train, X_test, Y_train, Y_test, plot=True):
        # Determine Labels for Training and Test Dat
        Y_pred_train = self.net.predict(X_train)

        m_train, intercept, r_value, p_value, std_err = stats.linregress(Y_train, Y_pred_train)
        Y_pred_test = self.net.predict(X_test)

        r_value, p_value, std_err  = self.Model_Class.plot(Y_train, Y_pred_train, Y_test, Y_pred_test, plot=plot)
        return Y_test, Y_pred_test, r_value

    def predict(self, X):
        # make predictions on unlabeled data
        return self.net.predict(X)

    def tune(self, X_train, Y_train):
        # define grid of possible hyper parameter values
        param_dist = {'nodes': sp_randint(5, 50),
                      'eta': sp_norm(.05, .1),
                      'lmbda': sp_uniform(0, 1),
                      'patience': sp_randint(5, 15)}

        self.net.set_params(verbose=False)
        random_search = RandomizedSearchCV(self.net,
            scoring='neg_mean_squared_error',
            param_distributions = param_dist, cv=5, n_jobs=8)
        random_search.fit(X_train, Y_train)
        self.params = random_search.best_params_
        self.net = Network(self.NF,
            nodes=self.params['nodes'],
            eta=self.params['eta'],
            lmbda=self.params['lmbda'],
            patience=self.params['patience'])
        return self.params

    def select_features(self, X_train, Y_train, keep='all', iterations=50):
        # perform feature ranking with subsets of training data

        # define percentage of training data to keep when bootstrapping
        p_bootstrap = .8
        NS, NF = X_train.shape

        if keep == 'all':
            keep = NF
        elif keep > NF:
            keep = NF

        self.net.set_params(verbose=False)
        def get_importances():
            # keep p_bootstrap % of training sample set after shuffling
            rand_inds = np.random.permutation(NS)
            X_train_sample = X_train[rand_inds, :][:int(p_bootstrap*NS), :]
            Y_train_sample = Y_train[rand_inds][:int(p_bootstrap*NS)]
            # fit NN model to bootstrap sample of training data
            self.net.fit(X_train_sample, Y_train_sample)
            importances = self.net.feature_importance(X_train_sample, Y_train_sample)
            self.reset()
            return importances
        feature_importances = Parallel(n_jobs=4)(delayed(get_importances)() for _ in range(iterations))

        # record mean feature importances
        mean_feature_importances = np.mean(feature_importances, 0)

        # return X_train with kept features
        selected_features = np.argsort(np.abs(mean_feature_importances))[::-1]
        sorted_feature_importances = mean_feature_importances[selected_features][:keep]
        X_train_select = X_train[:, selected_features[:keep]]

        # check if hyper parameters already optimized before re-instantiating
        try:
            self.NF = keep
            self.net = Network(self.NF,
                nodes=self.params['nodes'],
                eta=self.params['eta'],
                lmbda=self.params['lmbda'],
                verbose=True)
        except:
            self.NF = keep
            self.net = Network(self.NF, verbose=True)

        return X_train_select, selected_features[:keep], sorted_feature_importances

    def set_model_params(self, params):
        self.net.set_params(**params)

class BayesianLinearRegression:

    def __init__(self, Model_Class, X_train, Y_train):

        self.Model_Class = Model_Class

        NS_train, NF = X_train.shape
        self.X_train = X_train
        self.Y_train = Y_train
        self.blr = BLR(X_train, Y_train)

        if NS_train > NF:
            self.blr = BLR(X_train, Y_train)
        else:
            print("Warning: Number of features exceeds number of Samples")
            print("Try feature reduction first")

    def train(self, X_train, Y_train):

        self.blr = BLR(X_train, Y_train, a = 400, b = 3.0, max_evidence=True)
        print(self.blr.a)
        print(self.blr.b)

    def test(self, X_train, X_test, Y_train, Y_test, plot=True):
        # Determine Labels for Training and Test Data
        Y_pred_train, Y_pred_train_var = self.blr.predict(X_train)
        Y_pred_test, Y_pred_test_var = self.blr.predict(X_test)

        r_value, p_value, std_err = self.Model_Class.plot(Y_train, Y_pred_train,
            Y_test, Y_pred_test, Yerr=Y_pred_test_var.diagonal()**.5, plot=plot)
        return Y_test, Y_pred_test, r_value

    def predict(self, X):
        return self.blr.predict(X)

    def select_features(self, X_train, Y_train, keep='all'):

        NS_train, NF = X_train.shape
        NF_batch = NS_train
        N_mini_batches = NF // NF_batch
        weights = np.zeros(NF)
        for i in range(N_mini_batches):
            print('Running mini batch {0} of {1}'.format(i+1, N_mini_batches))
            self.blr = BLR(X_train[:, i*NF_batch:(i+1)*NF_batch], Y_train, a = 400, b = 3.0, max_evidence=True)
            weights[i*NF_batch:(i+1)*NF_batch] = self.blr.get_weights()

        if keep=='all' or keep > NF_batch:
            self.selected_features = np.argsort(np.abs(weights))[::-1][:NF_batch]
            feature_importances = sorted(abs(weights))[::-1]
        elif keep <= NF_batch:
            self.selected_features = np.argsort(np.abs(weights))[::-1][:keep]
            feature_importances = sorted(abs(weights))[::-1][:keep]

        return X_train[:, self.selected_features], self.selected_features, feature_importances

    def get_weights(self):
        return self.blr.get_weights()

class IndicatorSpecies:

    def __init__(self, OTU_table, targets):
        # import feature names and feature data
        self.OTU_table = OTU_table
        self.targets = targets
        self.feature_names = self.OTU_table.columns[1:]

        # import labels
        labels = pd.DataFrame()

        samples = self.OTU_table.iloc[:, 0]
        DOC = [self.targets[sample][0] for sample in samples]
        mean = np.mean(DOC)

        for sample in samples:
            if self.targets[sample][0] < mean:
                labels[sample] = ['Low']
            else:
                labels[sample] = ['High']
        self.labels = np.ravel(labels.values)

    def calc_IND_stat(self, labels):
        # get feature names and feature values from OTU_table
        feature_names = self.feature_names
        features = np.array(self.OTU_table.values[:, 1:], np.int)
        NS, NF = features.shape
        # ignore zero features
        inds = np.sum(features, 0) > 0
        # determine unique number of sites
        sites = np.unique(labels)
        indvals = np.zeros([NF, len(sites)])

        for i,label in enumerate(sites):
            # calculate A: measure of specificity, n_p / n
            # n_p_A: mean number of species in target site
            n_p_A = np.mean(features[labels==label, :], 0)
            # n: sum of mean number of species in each site
            n = 0
            for label_2 in sites:
                n += np.mean(features[labels==label_2, :], 0)
            A = n_p_A[inds] / n[inds]
            # Calculate B: measure of fidelity, n_p / N_p
            # n_p_B is the number of occurrences of species at target site group
            n_p_B = np.sum(features[labels==label, :] > 0, 0)
            N_p = np.sum(labels==label)
            B = n_p_B[inds] / N_p
            indvals[inds,i] = (A*B)**.5

        indicator_sites = np.array([sites[i] for i in np.argmax(indvals, 1)])
        indvals = np.max(indvals,1)

        return indvals, indicator_sites

    def calc_p(self, indval, nperm):
        def pv():
            randinds = np.random.permutation(len(self.labels))
            rand_labels = self.labels[randinds]
            temp_indval, site = self.calc_IND_stat(rand_labels)
            return temp_indval >= indval
        pvs = Parallel(n_jobs=4)(delayed(pv)() for _ in range(nperm))
        p_values = sum(pvs) / (1+nperm)
        return np.array(p_values)

    def run(self, max_p_value=None, save_data=False, sort_by=False, nperm=199):
        # initialize data frame to save file
        IS_results_dataframe = pd.DataFrame()
        # get indicator species stat for each feature
        indvals, indicator_sites = self.calc_IND_stat(self.labels)
        # calculate p values
        p_values = self.calc_p(indvals, nperm)
        # save data to dataframe
        IS_results_dataframe['OTUs'] = self.feature_names
        IS_results_dataframe['Stat'] = indvals
        IS_results_dataframe['Site Label'] = indicator_sites
        IS_results_dataframe['P value'] = p_values
        # remove insignificant indicator species
        if max_p_value:
            IS_results_dataframe = IS_results_dataframe.iloc[p_values<=max_p_value, :]

        # sort data frame by 'Stat'
        if sort_by:
            IS_results_dataframe = IS_results_dataframe.sort_values(by=sort_by, ascending=False)
        # save data to specified filename
        if save_data:
            IS_results_dataframe.to_csv(save_data, index=False)
        return IS_results_dataframe

class FeatureSelection:

    def __init__(self, train_data, test_data, targets, test_targets=None, rep_seqs=None):

        # include support for representative sequences information
        self.rep_seqs = rep_seqs

        if test_targets is None:
            test_targets = targets

        self.train_data = train_data
        self.test_data = test_data
        self.targets = targets
        self.test_targets = test_targets

    def FeatureSelectionTable(self, return_names=False, iterations=100):
        # method to perform FS from each method and generate table with results

        # create Indicator Species object
        IS = IndicatorSpecies(self.train_data, self.targets)
        IS_results = IS.run(sort_by='Stat')
        self.IS_results = IS_results

        # RF feature selection
        model = Model('Random Forest', self.train_data, self.test_data, self.targets, self.test_targets)
        train_data_RF, test_data_RF, names_RF, importances_RF = model.feature_selection(iterations=iterations)
        print(len(names_RF))
        # scale RF importances
        importances_RF /= np.max(importances_RF)
        self.names_RF = np.copy(names_RF)
        self.importances_RF = np.copy(importances_RF)

        # NN feature selection
        model = Model('Neural Network', self.train_data, self.test_data, self.targets, self.test_targets)
        train_data_NN, test_data, names_NN, importances_NN = model.feature_selection(iterations=iterations)
        # scale NN importances
        importances_NN /= np.max(np.abs(importances_NN))
        self.names_NN = np.copy(names_NN)
        self.importances_NN = np.copy(importances_NN)

        rf_fs_dict = {name:importance for name, importance in zip(names_RF, importances_RF)}
        nn_fs_dict = {name:importance for name, importance in zip(names_NN, importances_NN)}

        # only consider features with IS p value < .05
        IS_p_values_all = IS_results['P value'].values
        names_IS = IS_results['OTUs'].values[IS_p_values_all<=.05]
        IS_stat_all = IS_results['Stat'].values[IS_p_values_all<=.05]
        IS_site_labels_all  = IS_results['Site Label'].values[IS_p_values_all<=.05]
        IS_p_values_all = IS_p_values_all[IS_p_values_all<=.05]

        IS_stat_dict = {name:IS_stat for name, IS_stat in zip(names_IS, IS_stat_all)}
        IS_p_value_dict = {name:IS_p_value for name, IS_p_value in zip(names_IS, IS_p_values_all)}
        IS_site_label_dict = {name:IS_site_label for name, IS_site_label in zip(names_IS, IS_site_labels_all)}

        # only return otus from the consensus set
        N_keep = len(names_IS)
        all_otus = list(names_IS[:N_keep]) + list(names_NN[:N_keep]) + list(names_RF[:N_keep])
        unique, counts = np.unique(all_otus, return_counts=True)
        otus = unique[counts==3]

        # save results to Pandas dataframe
        results = pd.DataFrame()
        results['Taxa'] = otus

        rf_importance = [rf_fs_dict[otu] for otu in otus]
        results['RF Importance'] = rf_importance
        nn_importance = [nn_fs_dict[otu] for otu in otus]
        results['NN Importance'] = nn_importance
        IS_stat = [IS_stat_dict[otu] for otu in otus]
        results['IS stat'] = IS_stat
        IS_site_label = [IS_site_label_dict[otu] for otu in otus]
        results['IS Site Label'] = IS_site_label
        p_values = [IS_p_value_dict[otu] for otu in otus]
        results['IS P value'] = p_values

        # include rep_seqs info
        if self.rep_seqs is not None:

            # OTU names should be in first column
            rep_seq_otus = self.rep_seqs.values[:, 0]
            # Rep sequences should be in second column
            seqs = self.rep_seqs.values[:, 1]
            # make dictionary mapping otu to rep_seq
            rep_seq_dict = {otu:seq for otu,seq in zip(rep_seq_otus, seqs)}

            # get rep_seqs of selected taxa
            rep_seqs_selected = []
            # loop over RFINN selected taxa
            for otu in otus:
                # make sure that rep_seq info exists for taxa
                if otu in rep_seq_otus:
                    rep_seqs_selected.append(rep_seq_dict[otu])
                else:
                    rep_seqs_selected.append("N/A")

            # add rep_seq info to results table
            results["Rep. Sequences"] = rep_seqs_selected

        # sort results by IS stat
        results = results.sort_values(by='IS stat', ascending=False)
        self.results = results

        if return_names:
            return results, names_IS, names_RF, names_NN
        else:
            return results

    def plot(self, shared_OTUs, min_ftrs, max_ftrs, bin_size, tune):

        shared_OTUs = shared_OTUs[min_ftrs:max_ftrs]
        print("Calculating test performance...")
        r_vals_NN, r_vals_RF, N_features = self.get_r_vals(shared_OTUs, tune)
        N_ftrs_axis, _ = self.bin_samples(np.arange(min_ftrs, min_ftrs+len(shared_OTUs)), bin_size)
        mean_r_vals_NN, std_r_vals_NN = self.bin_samples(r_vals_NN, bin_size)
        mean_r_vals_RF, std_r_vals_RF = self.bin_samples(r_vals_RF, bin_size)

        plt.figure(figsize=(8, 5))
        plt.errorbar(N_ftrs_axis, mean_r_vals_RF, capsize=3, markersize=4, alpha=0.75, linestyle='none', marker='D', yerr = std_r_vals_RF, label="Random Forest")
        plt.errorbar(N_ftrs_axis, mean_r_vals_NN, capsize=3, markersize=4, alpha=0.75, linestyle='none', marker='D', yerr = std_r_vals_NN, label="Neural Network")
        #plt.axvline(x=len(joint_OTUs)
        plt.xlabel('Number of Features')
        plt.xticks(N_ftrs_axis)
        plt.ylabel('Prediction Performance')
        plt.legend(loc=4)
        plt.ylim([0, 1])
        #plt.savefig('figures/FS/PerformanceVsFeatures_Red.png', dpi=100)
        plt.show()

    def CompareFeatureSelection(self, model, compare_method, FS_table=None, min_ftrs=5, max_ftrs=100, bin_size=5, tune=False, fname=None):
        # first determine prediction performance using FS from the joint set
        if FS_table is not None:
            joint_OTUs = FS_table.iloc[:, 0].values[min_ftrs:max_ftrs+1]
        else:
            FS_table = self.FeatureSelectionTable()
            joint_OTUs = FS_table.iloc[:, 0].values[min_ftrs:max_ftrs+1]

        r_vals, N_features = self.get_r_vals_model(model, joint_OTUs, tune)
        N_ftrs_axis, _ = self.bin_samples(np.arange(min_ftrs, min_ftrs+len(joint_OTUs)), bin_size)
        mean_r_vals, std_r_vals = self.bin_samples(r_vals, bin_size)

        try:
            if compare_method == 'Random Forest':
                method_OTUs = self.names_RF[min_ftrs:max_ftrs+1]
            if compare_method == 'Neural Network':
                method_OTUs = self.names_NN[min_ftrs:max_ftrs+1]
        except:
            FSmodel = Model(compare_method, self.train_data, self.test_data, self.targets, self.test_targets)
            train_data_FS, test_data_FS, names, importances = FSmodel.feature_selection(iterations=100)
            method_OTUs = names[min_ftrs:max_ftrs]

        r_vals_method, N_features = self.get_r_vals_model(model, method_OTUs, tune)
        N_ftrs_axis_method, _ = self.bin_samples(np.arange(min_ftrs, min_ftrs+len(method_OTUs)), bin_size)
        mean_r_vals_method, std_r_vals_method = self.bin_samples(r_vals_method, bin_size)

        plt.figure(figsize=(8, 5))
        plt.errorbar(N_ftrs_axis, mean_r_vals, capsize=3, markersize=4, alpha=0.75, linestyle='none', marker='D', yerr = std_r_vals, label="Consensus features")
        plt.errorbar(N_ftrs_axis_method, mean_r_vals_method, capsize=3, markersize=4, alpha=0.75, linestyle='none', marker='D', yerr = std_r_vals_method, label="{} features".format(compare_method))
        #plt.axvline(x=len(joint_OTUs)
        plt.xlabel('Number of Features')
        #plt.xticks(N_ftrs_axis)
        plt.ylabel('Prediction Performance')
        plt.legend(loc=4)
        plt.ylim([0, 1])
        plt.title("{} prediction performance".format(model))
        if fname: plt.savefig(fname, dpi=300)
        plt.show()

    def FeatureSelectionScore(self, FS_table=None, method='Consensus', min_ftrs=5, max_ftrs=100, bin_size=5, tune=False):
        ''' function to plot the model performance on test data using feature
        selection results specified by a given method '''
        if method == 'Consensus':
            try:
                if FS_table is not None:
                    shared_OTUs = FS_table.iloc[:, 0].values
                else:
                    shared_OTUs = self.results.iloc[:, 0].values
            except:
                print("Running feature selection...")
                FS_results = self.FeatureSelectionTable()
                shared_OTUs = FS_results.iloc[:, 0].values
        if method == 'Indicator Species':
            try:
                IS_p_values_all = self.IS_results['P value'].values
                shared_OTUs = self.IS_results['OTUs'].values[IS_p_values_all<=.05]
            except:
                IS = IndicatorSpecies(self.train_data, self.targets)
                self.IS_results = IS.run(sort_by='Stat')
                IS_p_values_all = self.IS_results['P value'].values
                shared_OTUs = self.IS_results['OTUs'].values[IS_p_values_all<=.05]
        if method == 'Random Forest':
            try:
                shared_OTUs = self.names_RF
            except:
                model = Model(method, self.train_data, self.test_data, self.targets, self.test_targets)
                train_data_FS, test_data_FS, names, importances = model.feature_selection(iterations=100)
                shared_OTUs = names
        if method == 'Neural Network':
            try:
                shared_OTUs = self.names_NN
            except:
                model = Model(method, self.train_data, self.test_data, self.targets, self.test_targets)
                train_data_FS, test_data_FS, names, importances = model.feature_selection(iterations=100)
                shared_OTUs = names
        self.plot(shared_OTUs, min_ftrs, max_ftrs, bin_size, tune)

    def bin_samples(self, samples, bin_size):
        # define function to bin sets of n samples and return array with avg, std dev
        n = bin_size
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

    def get_r_vals(self, shared_OTUs, tune):
        # define function to get r values based on the list of OTUs
        # create model and make predictions with reduced feature set
        r_vals_RF = []
        r_vals_NN = []
        N_features = np.arange(len(shared_OTUs))

        for NF in N_features:
            NF = int(np.ceil(NF))
            #print("Testing with {} features".format(NF))
            train_data_FS = self.train_data[shared_OTUs[:NF+1]]
            train_data_FS.insert(0, 'Sample ID', self.train_data[self.train_data.columns[0]])

            test_data_FS = self.test_data[shared_OTUs[:NF+1]]
            test_data_FS.insert(0, 'Sample ID', self.test_data[self.test_data.columns[0]])

            # instantiate model class
            model = Model('Neural Network', train_data_FS, test_data_FS, self.targets, self.test_targets)
            if tune: model.tune_hyper_params()
            # train model
            model.train_model()
            # test model
            y_test, y_pred_test, r = model.test_model(plot=False)
            # save r values over range of number of selected features
            r_vals_NN.append(r)

            # instantiate model class
            model = Model('Random Forest', train_data_FS, test_data_FS, self.targets, self.test_targets)
            if tune: model.tune_hyper_params()
            # train model
            model.train_model()
            # test model
            y_test, y_pred_test, r = model.test_model(plot=False)
            # save r values over range of number of selected features
            r_vals_RF.append(r)

        return r_vals_NN, r_vals_RF, N_features

    def get_r_vals_model(self, model_name, shared_OTUs, tune):
        # define function to get r values based on the list of OTUs
        # create model and make predictions with reduced feature set
        r_vals = []
        N_features = np.arange(len(shared_OTUs))

        for NF in N_features:
            NF = int(np.ceil(NF))
            #print("Testing with {} features".format(NF))
            train_data_FS = self.train_data[shared_OTUs[:NF+1]]
            train_data_FS.insert(0, 'Sample ID', self.train_data[self.train_data.columns[0]])

            test_data_FS = self.test_data[shared_OTUs[:NF+1]]
            test_data_FS.insert(0, 'Sample ID', self.test_data[self.test_data.columns[0]])

            # instantiate model class
            model = Model(model_name, train_data_FS, test_data_FS, self.targets, self.test_targets)
            if tune: model.tune_hyper_params()
            # train model
            model.train_model()
            # test model
            y_test, y_pred_test, r = model.test_model(plot=False)
            # save r values over range of number of selected features
            r_vals.append(r)

        return r_vals, N_features


if __name__ == "__main__":
     # Example code to run program goes here:
     print("Running Random Forest DOC prediction...")

     # import training and test data as pandas dataframes
     train_data = pd.read_csv('train_data.csv')
     test_data = pd.read_csv('test_data.csv')
     targets = pd.read_csv('DOC_targets.csv')

     # instantiate model class
     model = Model('Random Forest', train_data, test_data, targets)

     # train model
     model.train_model()

     # test model
     model.test_model()

# Copyright 2020, gigo-gigo, All rights reserved.
# This code is based on the code by Yarin Gal used for his
# paper "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning".

import math
import random as python_random
import numpy as np
import argparse
import sys
import tensorflow as tf
from tensorflow.keras import backend as K

parser=argparse.ArgumentParser()

parser.add_argument('--dir', '-d', required=True, help='Name of the UCI Dataset directory. Eg: bostonHousing')
parser.add_argument('--epochx','-e', default=500, type=int, help='Multiplier for the number of epochs for training.')
parser.add_argument('--hidden', '-nh', default=2, type=int, help='Number of hidden layers for the neural net')

args=parser.parse_args()

data_directory = args.dir
epochs_multiplier = args.epochx
num_hidden_layers = args.hidden

sys.path.append('net/')

import net

# We delete previous results
import os


_RESULTS_VALIDATION_MC_LL = "./UCI_Datasets/" + data_directory + "/results/validation_MC_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_VP_LL = "./UCI_Datasets/" + data_directory + "/results/validation_VP_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_MC_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_MC_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_VP_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_VP_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

_RESULTS_TEST_MC10_LL = "./UCI_Datasets/" + data_directory + "/results/test_MC10_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_MC1000_LL = "./UCI_Datasets/" + data_directory + "/results/test_MC1000_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_VP_LL = "./UCI_Datasets/" + data_directory + "/results/test_VP_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_TAU = "./UCI_Datasets/" + data_directory + "/results/test_tau_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_RHO = "./UCI_Datasets/" + data_directory + "/results/test_rho_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_MC10_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_MC10_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_MC1000_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_MC1000_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_VP_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_VP_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_LOG = "./UCI_Datasets/" + data_directory + "/results/log_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

_DATA_DIRECTORY_PATH = "./UCI_Datasets/" + data_directory + "/data/"
_DROPOUT_RATES_FILE = _DATA_DIRECTORY_PATH + "dropout_rates.txt"
_TAU_VALUES_FILE = _DATA_DIRECTORY_PATH + "tau_values.txt"
_RHO_VALUES_FILE = _DATA_DIRECTORY_PATH + "rho_values.txt"
_DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
_HIDDEN_UNITS_FILE = _DATA_DIRECTORY_PATH + "n_hidden.txt"
_EPOCHS_FILE = _DATA_DIRECTORY_PATH + "n_epochs.txt"
_INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
_INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
_N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

def _get_index_train_test_path(split_num, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt" 


print ("Removing existing result files...")
if os.path.exists(_RESULTS_VALIDATION_MC_LL):
    os.remove(_RESULTS_VALIDATION_MC_LL)
if os.path.exists(_RESULTS_VALIDATION_VP_LL):
    os.remove(_RESULTS_VALIDATION_VP_LL)
if os.path.exists(_RESULTS_VALIDATION_RMSE):
    os.remove(_RESULTS_VALIDATION_RMSE)
if os.path.exists(_RESULTS_VALIDATION_MC_RMSE):
    os.remove(_RESULTS_VALIDATION_MC_RMSE)
if os.path.exists(_RESULTS_VALIDATION_VP_RMSE):
    os.remove(_RESULTS_VALIDATION_VP_RMSE)
if os.path.exists(_RESULTS_TEST_MC10_LL):
    os.remove(_RESULTS_TEST_MC10_LL)
if os.path.exists(_RESULTS_TEST_MC1000_LL):
    os.remove(_RESULTS_TEST_MC1000_LL)
if os.path.exists(_RESULTS_TEST_VP_LL):
    os.remove(_RESULTS_TEST_VP_LL)
if os.path.exists(_RESULTS_TEST_TAU):
    os.remove(_RESULTS_TEST_TAU)
if os.path.exists(_RESULTS_TEST_RHO):
    os.remove(_RESULTS_TEST_RHO)
if os.path.exists(_RESULTS_TEST_RMSE):
    os.remove(_RESULTS_TEST_RMSE)
if os.path.exists(_RESULTS_TEST_MC10_RMSE):
    os.remove(_RESULTS_TEST_MC10_RMSE)
if os.path.exists(_RESULTS_TEST_MC1000_RMSE):
    os.remove(_RESULTS_TEST_MC1000_RMSE)
if os.path.exists(_RESULTS_TEST_VP_RMSE):
    os.remove(_RESULTS_TEST_VP_RMSE)
if os.path.exists(_RESULTS_TEST_LOG):
    os.remove(_RESULTS_TEST_LOG)
print ("Result files removed.")

# We fix the random seed

seed = 1

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(seed)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(seed)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(seed)

print ("Loading data and other hyperparameters...")
# We load the data

data = np.loadtxt(_DATA_FILE)

# We load the number of hidden units

n_hidden = np.loadtxt(_HIDDEN_UNITS_FILE).tolist()

# We load the number of training epocs

n_epochs = np.loadtxt(_EPOCHS_FILE).tolist()

# We load the indexes for the features and for the target

index_features = np.loadtxt(_INDEX_FEATURES_FILE)
index_target = np.loadtxt(_INDEX_TARGET_FILE)

X = data[ : , [int(i) for i in index_features.tolist()] ]
y = data[ : , int(index_target.tolist()) ]

# We iterate over the training test splits

n_splits = np.loadtxt(_N_SPLITS_FILE)
print ("Done.")

errors, MC10_errors, MC10_lls, MC1000_errors, MC1000_lls, vp_errors, vp_lls = [], [], [], [], [], [], []
for split in range(int(n_splits)):

    # We load the indexes of the training and test sets
    print ('Loading file: ' + _get_index_train_test_path(split, train=True))
    print ('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]
    
    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    X_train_original = X_train
    y_train_original = y_train
    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:]
    X_train = X_train[0:num_training_examples, :]
    y_train = y_train[0:num_training_examples]
    
    # Printing the size of the training, validation and test sets
    print ('Number of training examples: ' + str(X_train.shape[0]))
    print ('Number of validation examples: ' + str(X_validation.shape[0]))
    print ('Number of test examples: ' + str(X_test.shape[0]))
    print ('Number of train_original examples: ' + str(X_train_original.shape[0]))

    # List of hyperparameters which we will try out using grid-search
    dropout_rates = np.loadtxt(_DROPOUT_RATES_FILE).tolist()
    tau_values = np.loadtxt(_TAU_VALUES_FILE).tolist()
    rho_values = np.loadtxt(_RHO_VALUES_FILE).tolist()

    # We perform grid-search to select the best hyperparameters based on the highest log-likelihood value
    best_network = None
    best_ll = -float('inf')
    best_tau = None
    best_dropout = None
    for dropout_rate in dropout_rates:
        for tau in tau_values:
            print ('Grid search step: Tau: ' + str(tau) + ' Dropout rate: ' + str(dropout_rate))
            K.clear_session()
            network = net.net(X_train, y_train, ([ int(n_hidden) ] * num_hidden_layers),
                    normalize = True, n_epochs = int(n_epochs * epochs_multiplier), tau = tau,
                    dropout = dropout_rate)

            # We obtain the test RMSE and the test ll from the validation sets

            error, MC10_error, MC10_ll, MC1000_error, MC1000_ll, vp_error, vp_ll = network.predict(X_validation, y_validation)
            if (MC1000_ll > best_ll):
                best_ll = MC1000_ll
                best_tau = tau
                best_dropout = dropout_rate
                print ('Best log_likelihood changed to: ' + str(best_ll))
                print ('Best tau changed to: ' + str(best_tau))
                print ('Best dropout rate changed to: ' + str(best_dropout))
            
            # Storing validation results
            with open(_RESULTS_VALIDATION_RMSE, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(error) + '\n')

            with open(_RESULTS_VALIDATION_MC_RMSE, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(MC1000_error) + '\n')

            with open(_RESULTS_VALIDATION_VP_RMSE, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(vp_error) + '\n')

            with open(_RESULTS_VALIDATION_MC_LL, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(MC1000_ll) + '\n')

            with open(_RESULTS_VALIDATION_VP_LL, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(vp_ll) + '\n')

    # Tuning rho
    K.clear_session()
    best_network = net.net(X_train, y_train, ([ int(n_hidden) ] * num_hidden_layers),
            normalize = True, n_epochs = int(n_epochs * epochs_multiplier), tau = best_tau,
            dropout = best_dropout)
    best_rho = best_network.tune_rho(X_validation, y_validation, rho_values)
    print ('Best rho: ' + str(best_rho))

    # Storing test results
    K.clear_session()
    best_network = net.net(X_train_original, y_train_original, ([ int(n_hidden) ] * num_hidden_layers),
                    normalize = True, n_epochs = int(n_epochs * epochs_multiplier), tau = best_tau,
                    dropout = best_dropout)
    error, MC10_error, MC10_ll, MC1000_error, MC1000_ll, vp_error, vp_ll = best_network.predict(X_test, y_test, best_rho)
    
    with open(_RESULTS_TEST_RMSE, "a") as myfile:
        myfile.write(repr(error) + '\n')

    with open(_RESULTS_TEST_MC10_RMSE, "a") as myfile:
        myfile.write(repr(MC10_error) + '\n')

    with open(_RESULTS_TEST_MC1000_RMSE, "a") as myfile:
        myfile.write(repr(MC1000_error) + '\n')

    with open(_RESULTS_TEST_VP_RMSE, "a") as myfile:
        myfile.write(repr(vp_error) + '\n')

    with open(_RESULTS_TEST_MC10_LL, "a") as myfile:
        myfile.write(repr(MC10_ll) + '\n')

    with open(_RESULTS_TEST_MC1000_LL, "a") as myfile:
        myfile.write(repr(MC1000_ll) + '\n')

    with open(_RESULTS_TEST_VP_LL, "a") as myfile:
        myfile.write(repr(vp_ll) + '\n')

    with open(_RESULTS_TEST_TAU, "a") as myfile:
        myfile.write(repr(best_tau) + '\n')

    with open(_RESULTS_TEST_RHO, "a") as myfile:
        myfile.write(repr(best_rho) + '\n')

    print ("Tests on split " + str(split) + " complete.")
    errors += [error]
    MC10_errors += [MC10_error]
    MC10_lls += [MC10_ll]
    MC1000_errors += [MC1000_error]
    MC1000_lls += [MC1000_ll]
    vp_errors += [vp_error]
    vp_lls += [vp_ll]

with open(_RESULTS_TEST_LOG, "a") as myfile:
    myfile.write('errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(errors), np.std(errors), np.std(errors)/math.sqrt(n_splits),
        np.percentile(errors, 50), np.percentile(errors, 25), np.percentile(errors, 75)))
    myfile.write('MC10 errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(MC10_errors), np.std(MC10_errors), np.std(MC10_errors)/math.sqrt(n_splits),
        np.percentile(MC10_errors, 50), np.percentile(MC10_errors, 25), np.percentile(MC10_errors, 75)))
    myfile.write('MC10 lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(MC10_lls), np.std(MC10_lls), np.std(MC10_lls)/math.sqrt(n_splits), 
        np.percentile(MC10_lls, 50), np.percentile(MC10_lls, 25), np.percentile(MC10_lls, 75)))
    myfile.write('MC1000 errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(MC1000_errors), np.std(MC1000_errors), np.std(MC1000_errors)/math.sqrt(n_splits),
        np.percentile(MC1000_errors, 50), np.percentile(MC1000_errors, 25), np.percentile(MC1000_errors, 75)))
    myfile.write('MC1000 lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(MC1000_lls), np.std(MC1000_lls), np.std(MC1000_lls)/math.sqrt(n_splits), 
        np.percentile(MC1000_lls, 50), np.percentile(MC1000_lls, 25), np.percentile(MC1000_lls, 75)))
    myfile.write('VP errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(vp_errors), np.std(vp_errors), np.std(vp_errors)/math.sqrt(n_splits),
        np.percentile(vp_errors, 50), np.percentile(vp_errors, 25), np.percentile(vp_errors, 75)))
    myfile.write('VP lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(vp_lls), np.std(vp_lls), np.std(vp_lls)/math.sqrt(n_splits), 
        np.percentile(vp_lls, 50), np.percentile(vp_lls, 25), np.percentile(vp_lls, 75)))

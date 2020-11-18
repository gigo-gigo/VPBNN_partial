import warnings
warnings.filterwarnings("ignore")

import math
from scipy.special import logsumexp
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import backend as K

import time

import sys
sys.path.append('../vpbnn_tf.keras/')
from vpbnn.models import nn2vpbnn

class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False, tau = 1.0, dropout = 0.05, T = 1000):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        self.var_y_train = np.var(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin = 2).T
        
        # We construct the network
        N = X_train.shape[0]
        batch_size = 128
        lengthscale = 1e-2
        reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(dropout)(inputs)
        inter = Dense(n_hidden[0], activation='relu', kernel_regularizer=l2(reg))(inter)
        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter)
            inter = Dense(n_hidden[i+1], activation='relu', kernel_regularizer=l2(reg))(inter)
        inter = Dropout(dropout)(inter)
        outputs = Dense(y_train_normalized.shape[1], kernel_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        model.compile(loss='mean_squared_error', optimizer='adam')

        # We iterate the learning process
        start_time = time.time()
        model.fit(X_train, y_train_normalized, batch_size=batch_size, epochs=n_epochs, verbose=0)
        self.model = model
        self.tau = tau
        self.T = T
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test, y_test, rho=0.0):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin = 2)
        y_test = np.array(y_test, ndmin = 2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        standard_pred = model.predict(X_test, batch_size=500)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5
        
        vmodel = nn2vpbnn(model, variance_mode=4)
        mcmodel = Model(inputs=vmodel.input, outputs=vmodel.output[0])
        Yt_hat = np.array([mcmodel.predict(X_test, batch_size=500) for _ in range(self.T)])
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train

        MC_pred = np.mean(Yt_hat[:10], 0)
        mc10_rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5
        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test - Yt_hat[:10])**2., 0) - np.log(10) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        mc10_test_ll = np.mean(ll)

        MC_pred = np.mean(Yt_hat[:1000], 0)
        mc1000_rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5
        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test - Yt_hat[:1000])**2., 0) - np.log(1000) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        mc1000_test_ll = np.mean(ll)        

        # https://github.com/janisgp/Sampling-free-Epistemic-Uncertainty/blob/master/UCI_regression/net/net.py#L172
        vmodel = nn2vpbnn(model, variance_mode=3, rho=rho)
        Y_mean, Y_var = vmodel.predict(X_test, batch_size=500)
        Y_mean = Y_mean * self.std_y_train + self.mean_y_train
        Y_var = Y_var * self.var_y_train
        Yt_hat = np.random.normal(loc=Y_mean, scale=np.sqrt(Y_var), size=(self.T, *Y_mean.shape))
        vp_rmse = np.mean((y_test.squeeze() - Y_mean.squeeze())**2.)**0.5

        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test - Yt_hat)**2., 0) - np.log(self.T) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        vp_test_ll = np.mean(ll)

        # We are done!
        return rmse_standard_pred, mc10_rmse, mc10_test_ll, mc1000_rmse, mc1000_test_ll, vp_rmse, vp_test_ll

    def tune_rho(self, X_val, y_val, rhos):

        X_val = np.array(X_val, ndmin = 2)
        y_val = np.array(y_val, ndmin = 2).T

        # We normalize the test set

        X_val = (X_val - np.full(X_val.shape, self.mean_X_train)) / \
            np.full(X_val.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        
        best_rho = None
        best_ll = -np.inf
        for rho in rhos:
            vmodel = nn2vpbnn(model, variance_mode=3, rho=rho)
            Y_mean, Y_var = vmodel.predict(X_val, batch_size=500)
            Y_mean = Y_mean * self.std_y_train + self.mean_y_train
            Y_var = Y_var * self.var_y_train
            Yt_hat = np.random.normal(loc=Y_mean, scale=np.sqrt(Y_var), size=(self.T, *Y_mean.shape))

            # We compute the test log-likelihood
            ll = (logsumexp(-0.5 * self.tau * (y_val - Yt_hat)**2., 0) - np.log(self.T) 
                - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
            ll = np.mean(ll)

            if ll > best_ll:
                best_rho = rho
                best_ll = ll

        return best_rho
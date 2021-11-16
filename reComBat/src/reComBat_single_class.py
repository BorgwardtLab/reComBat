"""
reComBat: reComBat_single_class.py

The reComBat algorithm implemented as a single class.
"""
# Author: Michael F. Adamer <mikeadamer@gmail.com>
# November 2021


import warnings
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

class reComBat(object):
    """
    reCombatClass

    Parameters
    ----------
    parametric : bool
        Choose a parametric or non-parametric empirical Bayes optimization.
        The default is True.
    model : str
        Choose a linear model, ridge, Lasso, elastic_net.
        The default is 'linear'.
    config : dict
        A dictionary containing kwargs for the model (see sklean.linear_model for details).
        The default is None.
    conv_criterion : float, optional
        The convergence criterion for the optimization.
        The default is 1e-4.
    max_iter : int, optional
        The maximum number of steps of the parametric empirical Bayes optimization.
        The detault is 1000.
    n_jobs : int, optional
        The number of parallel threads in the non-parametric optimization.
        If not given, then this is set to the number of cpus.
    mean_only : bool, optional
        Adjust the mean only. No scaling is performed.
        The default is False.
    optimize_params : bool, optional
        Perform empirical Bayes optimization.
        The default is True.
    reference_batch : str, optional
        Give a reference batch which is not adjusted.
        The default is None.
    verbose : bool, optional
        Enable verbose output.
        The default is True.
    """
    
    def __init__(self,
                 parametric=True,
                 model='linear',
                 config=None,
                 conv_criterion=1e-4,
                 max_iter=1000,
                 n_jobs=None,
                 mean_only = False,
                 optimize_params=True,
                 reference_batch = None,
                 verbose=True):
        self.parametric = parametric
        self.model = model
        if config is not None:
            self.config = config
        else:
            self.config={}
        self.conv_criterion = conv_criterion
        self.max_iter = max_iter
        if n_jobs is None:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs
        self.mean_only = mean_only
        self.optimize_params = optimize_params
        if reference_batch is not None:
            self.reference_batch = reference_batch
        else:
            self.reference_batch = None
        if verbose:
            logging.basicConfig(level=logging.INFO,format='[reComBat] %(asctime)s %(message)s')
        else:
            logging.disable(level=logging.CRITICAL)

        # Set to True to indicate that reComBat has been fitted.
        self.fit_tag = False

    def fit(self,data,batches,X=None,C=None):
        """
        Fit method.

        Parameters
        ----------
        data : pandas dataframe
            A pandas dataframe containing the data matrix.
            The format is (rows x columns) = (samples x features)
        batches : pandas series
            A pandas series containing the batch of each sample in the dataframe.
        X : pandas dataframe, optional
            The design matrix of the desired variation. Numerical columns should have '_numerical' as a postfix.
            The format is (rows x columns) = (samples x features) The default is None.
        C : pandas dataframe, optional
            The design matrix for the unwanted variation. Numerical columns should have '_numerical' as a postfix.
            The format is (rows x columns) = (samples x features). The default is None.

        Returns
        -------
        None
        """

        logging.info("Starting to fot reComBat.")
        if data.isna().any().any():
            raise ValueError("The data contains NaN values.")
        if batches.isna().any().any():
            raise ValueError("The batches contain NaN values.")

        unique_batches = batches.unique()
        num_batches = len(unique_batches)

        if num_batches == 1:
            raise ValueError("There should be at least two batches in the dataset.")

        batches_one_hot = pd.get_dummies(batches.astype(str)).values

        if np.any(batches_one_hot.sum(axis=0) == 1):
            raise ValueError("There should be at least two values for each batch.")

        if data.shape[0] != batches.shape[0]:
            raise ValueError("The batch and data matrix have a different number of samples.")

        if self.reference_batch is not None:
            self.reference_batch_idxs = np.where(batches.values==self.reference_batch)[0]
            self.reference_batch_numerical = np.where(pd.get_dummies(batches.astype(str)).columns==str(self.reference_batch))[0]
            if len(self.reference_batch_idxs) == 0:
                self.reference_batch = None
            batches_one_hot[:,self.reference_batch_numerical] = 1

        logging.info("Fit the linear model.")
        Z = self.fit_model_(data,batches_one_hot,X=X if X is not None else None,C=C if C is not None else None)

        if self.optimize_params:
            if self.parametric:
                logging.info("Starting the empirical parametric optimisation.")
                self.parametric_optimization_(Z,batches_one_hot)
                logging.info("Optimisation finished.")
            elif not self.parametric:
                logging.info("Starting the empirical non-parametric optimisation.")
                self.non_parametric_optimization_(Z,batches_one_hot)
                logging.info("Optimisation finished.")
        else:
            logging.info("Compute parameters without optimising.")
            self.gamma_star_hat_, self.delta_star_squared_hat_ = self.compute_values_non_parametric_(Z,batches_one_hot)
            if self.reference_batch is not None:
                self.gamma_star_hat_[self.reference_batch_numerical] = 1
                self.delta_star_squared_hat_[self.reference_batch_numerical] = 1

        self.fit_tag = True
        logging.info("reComBat is fitted.")

    def transform(self,data,batches,X=None,C=None):
        """
        Transform method.
        ----------------

        Adjusts a dataframe. Please make sure that the number of batches,
        features and design matrix features match.

        Parameters
        ----------
        data : pandas dataframe
            A pandas dataframe containing the data matrix.
            The format is (rows x columns) = (samples x features)
        batches : pandas series
            A pandas series containing the batch of each sample in the dataframe.
        X : pandas dataframe, optional
            The design matrix of the desired variation. Numerical columns should have '_numerical' as a postfix.
            The format is (rows x columns) = (samples x features) The default is None.
        C : pandas dataframe, optional
            The design matrix for the unwanted variation. Numerical columns should have '_numerical' as a postfix.
            The format is (rows x columns) = (samples x features). The default is None.

        Returns
        -------
        A pandas dataframe of the same shape as the input dataframe.
        """

        logging.info("Starting to transform.")
        batches_one_hot = pd.get_dummies(batches.astype(str)).values

        if X is not None:
            X_categorical = X.loc[:,[c for c in X.columns if '_numerical' not in c]]
            X_numerical = X.loc[:,[c for c in X.columns if '_numerical' in c]].values
            X_categorical_one_hot = pd.get_dummies(X_categorical.astype(str),drop_first=True).values
            X_covariates = np.hstack([X_categorical_one_hot,X_numerical])
        if C is not None:
            C_categorical = C.loc[:,[c for c in X.columns if '_numerical' not in c]]
            C_numerical = C.loc[:,[c for c in X.columns if '_numerical' in c]].values
            C_categorical_one_hot = pd.get_dummies(C_categorical.astype(str),drop_first=True).values
            C_covariates = np.hstack([C_categorical_one_hot,C_numerical])

        if not self.fit_tag:
            raise AttributeError("reComBat has not been fitted yet.")
        if data.shape[1] != self.alpha_.shape[1]:
                raise ValueError("Wrong number of features.")
        if batches_one_hot.shape[1] != self.gamma_star_hat_.shape[0]:
            raise ValueError("Wrong number of batches.")
        if X is not None:
            if X_covariates.shape[1] != self.beta_x_.shape[0]:
                raise ValueError("The feature dimensions of fit X design matrix and transform X design matrix are different.")
            else:
                X_tmp = np.matmul(X_covariates,self.beta_x_)
        else:
            X_tmp = 0
        if C is not None:
            if C_covariates.shape[1] != self.beta_c_.shape[0]:
                raise ValueError("The feature dimensions of fit C design matrix and transform C design matrix are different.")
            else:
                C_tmp = np.matmul(X_covariates,self.beta_x_)
        else:
            C_tmp = 0

        Z = (data.values.copy() - self.alpha_ - X_tmp - C_tmp)/np.sqrt(self.sigma_)

        data_adjusted = self.adjust_data_(Z,batches_one_hot,X_covariates=X_covariates if X is not None else None)

        if self.reference_batch is not None:
            data_adjusted[self.reference_batch_idxs] = data.values[self.reference_batch_idxs]

        logging.info("Transform finished.")

        return pd.DataFrame(data_adjusted,index=data.index,columns=data.columns)

    def fit_transform(self,data,batches,X=None,C=None):
        '''
        Fit and transform in one go.
        '''
        self.fit(data,batches,X=X if X is not None else None, C=C if C is not None else None)
        return self.transform(data,batches,X=X if X is not None else None, C=C if C is not None else None)

    def fit_model_(self,data,batches_one_hot,X=None,C=None):
        '''
        Fit the linear model.
        '''

        # Create the design matrix
        num_batches = batches_one_hot.shape[1]
        Covariates = batches_one_hot

        if X is not None:
            if X.isna().any().any():
                raise ValueError("The design matrix X contains NaN values.")
            X_categorical = X.loc[:,[c for c in X.columns if '_numerical' not in c]]
            X_numerical = X.loc[:,[c for c in X.columns if '_numerical' in c]].values
            X_categorical_one_hot = pd.get_dummies(X_categorical.astype(str),drop_first=True).values
            X_covariates = np.hstack([X_categorical_one_hot,X_numerical])
            X_covariates_dim = X_covariates.shape[1]
            Covariates = np.hstack([Covariates,X_covariates])
        else:
            X_covariates_dim = 0
        if C is not None:
            if C.isna().any().any():
                raise ValueError("The design matrix C contains NaN values.")
            C_categorical = C.loc[:,[c for c in X.columns if '_numerical' not in c]]
            C_numerical = C.loc[:,[c for c in X.columns if '_numerical' in c]].values
            C_categorical_one_hot = pd.get_dummies(C_categorical.astype(str),drop_first=True).values
            C_covariates = np.hstack([C_categorical_one_hot,C_numerical])
            C_covariates_dim = C_covariates.shape[1]
            Covariates = np.hstack([Covariates,C_covariates])
            Covariates = Covariates.astype(float)
        else:
            C_covariates_dim = 0

        # Initialise the model class
        if self.model == 'linear':
            model = LinearRegression(fit_intercept=False,**self.config)
        elif self.model == 'ridge':
            model = Ridge(fit_intercept=False,**self.config)
        elif self.model == 'Lasso':
            model = Lasso(fit_intercept=False,**self.config)
        elif self.model == 'elastic_net':
            model = ElasticNet(fit_intercept=False,**self.config)
        else:
            raise ValueError('Model not implemented')

        model.fit(Covariates,data.values)

        # Save the fitted parameters
        # Note that alpha is computed implicitly via the contraints on the batch parameters.
        if self.reference_batch is None:
            self.alpha_ = np.matmul(batches_one_hot.sum(axis=0,keepdims=True)/batches_one_hot.sum(),model.coef_.T[:num_batches])
        else:
            self.alpha_ = model.coef_.T[self.reference_batch_numerical]
        self.beta_x_ = model.coef_.T[num_batches:num_batches+X_covariates_dim]
        self.beta_c_ = model.coef_.T[num_batches+X_covariates_dim:]

        # Compute the standard deviation of the reconstructed data.
        data_hat = np.matmul(Covariates, model.coef_.T)
        if self.reference_batch is None:
            self.sigma_ = np.mean((data.values - data_hat)**2,axis=0,keepdims=True)
        else:
            self.sigma_ = np.mean((data.values[self.reference_batch_idxs]-data_hat[self.reference_batch_idxs])**2,axis=0,keepdims=True)

        # Standardise the data.
        Z = (data.values.copy() - np.matmul(Covariates[:, num_batches:num_batches+X_covariates_dim], self.beta_x_)\
                        - np.matmul(Covariates[:, num_batches+X_covariates_dim:], self.beta_c_)\
                        - self.alpha_)/np.sqrt(self.sigma_)
        return Z

    def compute_init_values_parametric_(self,Z,batches_one_hot):
        '''
        Compute the starting values of the Bayesian optimization.
        '''
        gamma_hat = np.array([np.mean(Z[batches_one_hot[:,i]==1],axis=0) for i in range(batches_one_hot.shape[1])])
        gamma_bar = np.mean(gamma_hat,axis=1)
        tau_bar_squared = np.var(gamma_hat,axis=1,ddof=1)
        delta_hat_squared = np.array([np.var(Z[batches_one_hot[:,i]==1],axis=0,ddof=1) for i in range(batches_one_hot.shape[1])])
        V_bar = np.mean(delta_hat_squared,axis=1)
        S_bar_squared = np.var(delta_hat_squared,axis=1,ddof=1)

        lambda_bar = (V_bar**2+2*S_bar_squared)/S_bar_squared
        theta_bar = (V_bar**3+V_bar*S_bar_squared)/S_bar_squared

        return gamma_hat,gamma_bar,delta_hat_squared,tau_bar_squared,lambda_bar,theta_bar

    def compute_values_non_parametric_(self,Z,batches_one_hot):
        '''
        Compute the starting values of the Bayesian optimization.
        '''
        gamma_hat = np.array([np.mean(Z[batches_one_hot[:,i]==1],axis=0) for i in range(batches_one_hot.shape[1])])
        delta_hat_squared = np.array([np.var(Z[batches_one_hot[:,i]==1],axis=0,ddof=1) for i in range(batches_one_hot.shape[1])])
        return gamma_hat,delta_hat_squared

    def compute_weights_(self,Z_i,gamma_hat_i,delta_hat_squared_i):
        '''
        Compute the weights w_{ig} of the non-parametric Bayesian optimization.
        '''
        out = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_weights_util_)(g,Z_i[:,g],gamma_hat_i,delta_hat_squared_i,Z_i.shape[1]) for g in range(Z_i.shape[1]))
        return np.vstack(out)

    def compute_weights_util_(self,g,Z_ig,gamma_hat_i,delta_hat_squared_i,n_genes):
        # Always delete the current gene as the parameters are confounded.
        tmp = (Z_ig.reshape(-1,1) - np.delete(gamma_hat_i,g))**2/np.delete(delta_hat_squared_i,g)
        return np.prod((1/np.sqrt(2*np.pi*np.delete(delta_hat_squared_i,g)))*np.exp(-0.5*tmp),axis=0)

    def new_gamma_parametric_(self,n_i,tau_bar_squared_i,gamma_hat_i,delta_star_squared_i,gamma_bar_i):
        numerator = n_i*tau_bar_squared_i*gamma_hat_i + delta_star_squared_i*gamma_bar_i
        denominator = n_i*tau_bar_squared_i + delta_star_squared_i
        return numerator/denominator

    def new_delta_star_squared_parametric_(self, theta_bar_i,Z_i,gamma_star_new_i,n_i,lambda_bar_i):
        numerator = (theta_bar_i+0.5*np.sum((Z_i-gamma_star_new_i)**2,axis=0))
        denominator = (0.5*n_i+lambda_bar_i-1)
        return numerator/denominator

    def parametric_update_(self,
                           gamma_star_i,
                           delta_star_squared_i,
                           n_i,
                           tau_bar_squared_i,
                           gamma_hat_i,
                           gamma_bar_i,
                           theta_bar_i,
                           lambda_bar_i,
                           Z_i):
        '''
        Perform the optimization for one batch at a time
        '''
        gamma_star_new_i = gamma_star_i.copy()
        delta_star_squared_new_i = delta_star_squared_i.copy()

        iterations = 0
        convergence = self.conv_criterion + 1
        while (convergence > self.conv_criterion) and (iterations < self.max_iter):
            gamma_star_new_i = self.new_gamma_parametric_(n_i,
                                                          tau_bar_squared_i,
                                                          gamma_hat_i,
                                                          delta_star_squared_i,
                                                          gamma_bar_i)
            delta_star_squared_new_i = self.new_delta_star_squared_parametric_(theta_bar_i,
                                                                               Z_i,
                                                                               gamma_star_new_i,
                                                                               n_i,
                                                                               lambda_bar_i)

            convergence =np.max([np.max(np.abs(gamma_star_new_i-gamma_star_i)/gamma_star_i),\
                                np.max(np.abs(delta_star_squared_new_i-delta_star_squared_i)/delta_star_squared_i)])
            gamma_star_i = gamma_star_new_i
            delta_star_squared_i = delta_star_squared_new_i
            iterations += 1
        if (iterations >= self.max_iter):
            warnings.warn("Maximum number of iterations reached", RuntimeWarning)
        return gamma_star_i,delta_star_squared_i

    def parametric_optimization_(self,Z,batches_one_hot):
        '''
        Perform parametric optimization.
        '''
        gamma_hat,gamma_bar,delta_hat_squared,tau_bar_squared,lambda_bar,theta_bar = self.compute_init_values_parametric_(Z,batches_one_hot)
        n = batches_one_hot.sum(axis=0)

        gamma_star = gamma_hat.copy()
        delta_star_squared = delta_hat_squared.copy()

        for i in range(batches_one_hot.shape[1]):
            if not self.mean_only:
                gamma_star[i],delta_star_squared[i] = self.parametric_update_(gamma_star[i],
                                                                        delta_star_squared[i],
                                                                        n[i],
                                                                        tau_bar_squared[i],
                                                                        gamma_hat[i],
                                                                        gamma_bar[i],
                                                                        theta_bar[i],
                                                                        lambda_bar[i],
                                                                        Z[batches_one_hot[:,i]==1])
            else:
                gamma_star[i] = self.new_gamma_parametric_(1,tau_bar_squared[i],gamma_hat[i],1,gamma_bar[i])
                delta_star_squared[i] = np.ones_like(delta_hat_squared[i])
        self.gamma_star_hat_ = gamma_star
        self.delta_star_squared_hat_ = delta_star_squared

        if self.reference_batch is not None:
            self.gamma_star_hat_[self.reference_batch_numerical] = 1
            self.delta_star_squared_hat_[self.reference_batch_numerical] = 1

    def non_parametric_optimization_(self,Z,batches_one_hot):
        '''
        Perform non-parametric optimization.
        '''
        gamma_hat,delta_hat_squared = self.compute_values_non_parametric_(Z,batches_one_hot)

        gamma_star = np.zeros((batches_one_hot.shape[1],Z.shape[1]))
        delta_star_squared = np.zeros((batches_one_hot.shape[1],Z.shape[1]))

        for i in tqdm(range(batches_one_hot.shape[1])):
            if self.mean_only:
                delta_hat_squared[i] = np.ones_like(delta_hat_squared[i])

            weights = self.compute_weights_(Z[batches_one_hot[:,i]==1],gamma_hat[i],delta_hat_squared[i])

            gamma_star_numerator = np.vstack([weights[j]*np.delete(gamma_hat[i],j) for j in range(Z.shape[1])])
            gamma_star[i] = (np.sum(gamma_star_numerator,axis=1)/np.sum(weights,axis=1)).T

            delta_star_numerator = np.vstack([weights[j]*np.delete(delta_hat_squared[i],j) for j in range(Z.shape[1])])
            delta_star_squared[i] = (np.sum(delta_star_numerator,axis=1)/np.sum(weights,axis=1)).T

        self.gamma_star_hat_ = gamma_star
        self.delta_star_squared_hat_ = delta_star_squared

        if self.reference_batch is not None:
            self.gamma_star_hat_[self.reference_batch_numerical] = 1
            self.delta_star_squared_hat_[self.reference_batch_numerical] = 1

    def adjust_data_(self,Z,batches_one_hot,X_covariates=None):
        '''
        Perform the final adjustment step.
        '''
        tmp = np.zeros_like(Z)
        for i in range(batches_one_hot.shape [1]):
            tmp[batches_one_hot[:,i]==1] = (Z[batches_one_hot[:,i]==1]-self.gamma_star_hat_[i])/np.sqrt(self.delta_star_squared_hat_[i])
        data_adjusted = np.sqrt(self.sigma_)*tmp + self.alpha_
        if X_covariates is not None:
            data_adjusted += np.matmul(X_covariates,self.beta_x_)
        return data_adjusted

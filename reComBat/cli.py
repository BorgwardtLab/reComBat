"""
reComBat: cli.py

Command line interface.
This is the command line interface of the reComBat package.

You can run the package from the terminal via: reComBat --args.
The arguments are any arguments you can give to reComBat or sklearn linear_model,ridge,lasso or elastic_net.

The inputs should be *.csv files of shape (rows x columns) = (samples x features or batches).
The sample index is column 0.
"""
# Author: Michael F. Adamer <mikeadamer@gmail.com>
# November 2021

import os
from fire import Fire
import pandas as pd
from multiprocessing import cpu_count
from .reComBat import reComBat

def run_recombat(data_file,
                 batch_file,
                 X_file=None,
                 C_file=None,
                 data_path='.',
                 out_path='.',
                 out_file='data_adjusted.csv',
                 parametric=True,
                 model='linear',
                 conv_criterion=1e-4,
                 max_iter_eb=1000,
                 n_jobs=cpu_count(),
                 mean_only = False,
                 optimize_params=True,
                 reference_batch = None,
                 verbose=True,
                 alpha=None,
                 l1_ratio=None,
                 precompute=None,
                 max_iter=None,
                 copy_X=None,
                 tol=None,
                 warm_start=None,
                 positive=None,
                 random_state=None,
                 selection=None,
                 solver=None):

    # Read in the data and batch files.
    data = pd.read_csv(os.path.join(data_path,data_file),index_col=0)
    batch = pd.read_csv(os.path.join(data_path,batch_file),index_col=0,squeeze=True)

    # Read in potential design matrices.
    if X_file is not None:
        X = pd.read_csv(os.path.join(data_path,X_file),index_col=0)
    else:
        X = None

    if C_file is not None:
        C = pd.read_csv(os.path.join(data_path,C_file),index_col=0)
    else:
        C = None

    # Define the config dictionary. These are all arguments passed to sklearn.
    config = {}

    if alpha is not None:
        config['alpha'] = float(alpha)
    if l1_ratio is not None:
        config['l1_ratio'] = float(l1_ratio)
    if precompute is not None:
        config['precompute'] = precompute
    if max_iter is not None:
        config['max_iter'] = int(max_iter)
    if copy_X is not None:
        config['copy_X'] = copy_X
    if tol is not None:
        config['tol'] = float(tol)
    if warm_start is not None:
        config['warm_start'] = warm_start
    if positive is not None:
        config['positive'] = positive
    if random_state is not None:
        config['random_state'] = int(random_state)
    if selection is not None:
        config['selection'] = selection
    if solver is not None:
        config['solver'] = solver

    # Init reComBat.
    combat = reComBat(parametric=parametric,
                      model=model,
                      config=config,
                      conv_criterion=conv_criterion,
                      max_iter=max_iter_eb,
                      n_jobs=n_jobs,
                      mean_only=mean_only,
                      optimize_params=optimize_params,
                      reference_batch=reference_batch,
                      verbose=verbose)

    # Adjust data.
    data_adjusted = combat.fit_transform(data,batch,X=X,C=C)

    # Save adjusted data.
    data_adjusted.to_csv(os.path.join(out_path,out_file),index=True)

def main():
    Fire(run_recombat)

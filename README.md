# reComBat


[![License: BSD](https://img.shields.io/github/license/BorgwardtLab/recombat)](https://opensource.org/licenses/BSD-3-Clause)
[![Version](https://img.shields.io/pypi/v/recombat)](https://pypi.org/project/recombat/)
[![PythonVersion](https://img.shields.io/pypi/pyversions/recombat)]()

This is the reComBat implementation as described in our [recent paper](https://doi.org/10.1101/2021.11.22.469488).
The paper introduces a generalized version of the empirical Bayes batch correction method introduced in [1].
We use the two-design-matrix approach of Wachinger et al. [2]


## Installation

reComBat is a PyPI package which can be installed via `pip`:

```
pip install reComBat
```

You can also clone the repository and install it locally via [Poetry](https://python-poetry.org/) by executing
```bash
poetry install
```
in the repository directory.

## Usage

The `reComBat` package is inspired by the code of [3] and also uses a scikit-learn like
API.

In a Python script, you can import it via
```python
from reComBat import reComBat

combat = reComBat()
combat.fit(data,batches)
combat.transform(data,batches)
```
or

```python
combat.fit_transform(data,batches)
```

All data input (data, batches, design matrices) are input as pandas dataframes.
The format is (rows x columns) = (samples x features), and the index is an arbitrary sample index.
The batches should be given as a pandas series. Note that there are two types of columns for design matrices,
numerical columns and categorical columns. All columns in X and C are by default assumed categorical. If a column contains numerical
covariates, these columns should have the suffix "_numerical" in the column name.

There is also a command-line interface which can be called from a bash shell.
```bash
reComBat data_file.csv batch_file.csv --<optional args>
```

## Arguments

The `reComBat` class has many optional arguments (see below).
The `fit`, `transform` and `fit_transform` functions all take pandas dataframes as arguments,
`data` and `batches`. Both dataframes should be in the form above.

## Optional arguments

The `reComBat` class has the following optional arguments:

  - `parametric` : `True` or `False`. Choose between the parametric or non-parametric version of the empirical Bayes method.
  By default, this is `True`, i.e. the parametric method is performed. Note that the non-parametric method has a longer run time than the parametric one.
  - `model` : Choose which regression model should be used to standardise the data. You can choose between `linear`, `ride`, `lasso` and `elastic_net` regression.
  By default the `elastic_net` model is used.
  - `config` : A Python dictionary specifying the keyword arguments for the relevant `scikit-learn` regression functions. for further details refer to [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model). The default is `None`.
  - `conv_criterion` : The convergence criterion for the parametric empirical Bayes optimization. Relative, rather than absolute convergence criteria are used.
  The default is 1e-4.
  - `max_iter` : The maximum number of iterations for the parametric empirical Bayes optimization. The default is 1000.
  - `n_jobs` : The number of parallel thread used in the non-parametric empirical Bayes optimization. A larger number of threads considerably speeds up the computation, but also has higher memory requirements. The default is the number of CPUs of the machine.
  - `mean_only` : `True` or `False`.  Chooses whether the only the means are adjusted (no scaling is performed), or the full algorithm should be run. The default is `False`.
  - `optimize_params` : `True` or `False`. Chooses whether the Bayesian parameters should be optimised, or if the starting values should be used. The default is `True`.
  - `reference_batch` : If the data contains a reference batch, then this can be specified here. The reference batch will not be adjusted. The default is `None`.
  - `verbose` : `True` or `False`. Toggles verbose output. The default is `True`.

The command line interface can take any of these arguments (except for `config`) via `--<argument>=ARG`. Any `scikit-learn` keyword arguments should be given explicitly, e.g. `--alpha=1e-10`. The command line interface has the additional following optional arguments:
  - `X_file` : The csv file containing the design matrix of desired variation. The default is `None`.
  - `C_file` : The csv file containing the design matrix of undesired variation. The default is `None`.
  - `data_path` : The path to the data/design matrices. The default is the current directory.
  - `out_path` : The path where the output file should be stored. The default is the current directory.
  - `out_file` : The name out the output file (with extension).

## Output

The `transform` method and the command line interface output a dataframe, respectively a csv file, of the form (samples x features) with the adjusted data.

## Tutorial

We included a step-by-step tutorial in the `tutorial` folder of the GitHub repository. We also provide a PDF version which serves as a manual.

## Contact

This code is developed and maintained by members of the [Machine Learning and
Computational Biology Lab](https://www.bsse.ethz.ch/mlcb) of [Prof. Dr.
Karsten Borgwardt](https://www.bsse.ethz.ch/mlcb/karsten.html):

- [Michael Adamer](https://mikeadamer.github.io/) ([GitHub](https://github.com/MikeAdamer))
- Sarah Brüningk ([GitHub](https://github.com/sbrueningk))

*References*:

[1] W. Evan Johnson, Cheng Li, Ariel Rabinovic, Adjusting batch effects in microarray expression data using empirical Bayes methods, Biostatistics, Volume 8, Issue 1, January 2007, Pages 118–127, https://doi.org/10.1093/biostatistics/kxj037


[2] Christian Wachinger, Anna Rieckmann, Sebastian Pölsterl. Detect and Correct Bias in Multi-Site Neuroimaging Datasets. arXiv:2002.05049

[3] pycombat, CoAxLab, https://github.com/CoAxLab/pycombat

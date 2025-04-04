# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pytensor.tensor as pt
from scipy.stats import gamma


#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

np.random.seed(27) # set numpy random seed

# read data
paths = glob.glob("./chunked_data/*")
paths = np.sort(paths)

dfs_sorted = [pd.read_csv(paths[i]) for i in range(len(paths))]

df_all = pd.concat(dfs_sorted)

n_obs = np.array([len(df) for df in dfs_sorted])

# Factorize SiteIdx to create integer indices
idxs = pd.factorize(df_all.site_id)[0]
sites_id = pd.unique(idxs)
n_sites = len(sites_id)

# Extract values from the DataFrame
lower = df_all.min_incubation.values
upper = df_all.max_incubation.values

##No exact data was simulated
interval_indices = np.array([i for i, (x, y) in enumerate(zip(lower, upper)) if x != y])

interval = np.array([[lower[i], upper[i]] for i in interval_indices])

interval = np.array([[x,y] for x,y in zip(lower, upper)])

# Extract values from the DataFrame
lower = df_all.min_incubation.values
upper = df_all.max_incubation.values

# Separate exact and inexact intervals for left and right bounds
# Keep track of the original indices for alignment
interval_indices = np.array([i for i, (x, y) in enumerate(zip(lower, upper)) if x != y])
exact_indices = np.array([i for i, (x, y) in enumerate(zip(lower, upper)) if x == y])

# Extract data for exact and inexact intervals
interval = np.array([[lower[i], upper[i]] for i in interval_indices])
exact = np.array([lower[i] for i in exact_indices])

# Extract the corresponding indices for exact and inexact data
interval_idxs = idxs[interval_indices]
exact_idxs = idxs[exact_indices]


# Define the Gamma CDF and censored likelihood

def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)

weights = pt.sqrt(1/n_obs)

# Build the PyMC model
with pm.Model() as mod:
    
    alpha_s = pm.HalfNormal("alpha_s", 0.5)
    beta_s = pm.HalfNormal("beta_s", 0.5)
    
    alpha_z = pm.Normal('alpha_z', 0, 0.1, shape=n_sites) 
    beta_z = pm.Normal('beta_z', 0, 0.1, shape=n_sites)
    
    # Transformed parameters (ensuring positivity)
    alpha = pm.Deterministic("alpha", pt.exp(pt.log(7) + alpha_s * alpha_z))  # exp transform
    beta = pm.Deterministic("beta", pt.exp(pt.log(0.9) + beta_s * beta_z))     # exp transform
    
    mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
    
    # Latent likelihood for inexact intervals 
    w = pm.Potential('w', censored('censored', alpha[interval_idxs], beta[interval_idxs], 
                                     interval[:,0], interval[:,1]))
    
    # likelihood for exact intervals 
    y = pm.Gamma('y', alpha=alpha[exact_idxs], beta=beta[exact_idxs], observed=exact)
    
# Sample from the model
with mod:
    idata_dir = pm.sample(2000, tune=2000, nuts_sampler='numpyro', random_seed=27, target_accept=0.95)
    
try:
    del idata_dir.observed_data
except:
    pass
try:
    del idata_dir.sample_stats
except:
    pass

az.to_netcdf(idata_dir, "./direct_analysis_idata.nc")

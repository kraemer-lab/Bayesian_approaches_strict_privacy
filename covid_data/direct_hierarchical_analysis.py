# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pytensor.tensor as pt


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
idxs = pd.factorize(df_all.SiteIdx)[0]
sites_id = pd.unique(idxs)
n_sites = len(sites_id)

# Extract values from the DataFrame
left_lower = df_all.left_lower.values
left_upper = df_all.left_upper.values
right_lower = df_all.right_lower.values
right_upper = df_all.right_upper.values

# Separate exact and inexact intervals for left and right bounds
# Keep track of the original indices for alignment
left_interval_indices = np.array([i for i, (x, y) in enumerate(zip(left_lower, right_lower)) if x != y])
left_exact_indices = np.array([i for i, (x, y) in enumerate(zip(left_lower, right_lower)) if x == y])

right_interval_indices = np.array([i for i, (x, y) in enumerate(zip(left_upper, right_upper)) if x != y])
right_exact_indices = np.array([i for i, (x, y) in enumerate(zip(left_upper, right_upper)) if x == y])

# Extract data for exact and inexact intervals
left_interval = np.array([[left_lower[i], right_lower[i]] for i in left_interval_indices])
left_exact = np.array([left_lower[i] for i in left_exact_indices])

right_interval = np.array([[left_upper[i], right_upper[i]] for i in right_interval_indices])
right_exact = np.array([left_upper[i] for i in right_exact_indices])

# Extract the corresponding indices for exact and inexact data
left_interval_idxs = idxs[left_interval_indices]
left_exact_idxs = idxs[left_exact_indices]

right_interval_idxs = idxs[right_interval_indices]
right_exact_idxs = idxs[right_exact_indices]

# ICG likelihood function
def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)


# Build the PyMC model
with pm.Model() as mod:
        
    alpha_s = pm.HalfNormal("alpha_s", 0.5)
    beta_s = pm.HalfNormal("beta_s", 0.5)
    
    alpha_z = pm.Normal('alpha_z', 0, 0.1, shape=n_sites) 
    beta_z = pm.Normal('beta_z', 0, 0.1, shape=n_sites)
    
    # Transformed parameters (ensuring positivity)
    alpha = pm.Deterministic("alpha",  pt.exp(pt.log(4) + alpha_s * alpha_z))  # exp transform
    beta = pm.Deterministic("beta",  pt.exp(pt.log(0.66) + beta_s * beta_z))     # exp transform
   
    mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
    
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
            
    # Latent likelihood for inexact intervals (left)
    wl = pm.Potential('wl', censored('censoredl', alpha[left_interval_idxs], beta[left_interval_idxs], 
                                     left_interval[:, 1], left_interval[:, 0]))
    
    # Likelihood for exact intervals (left)
    yl = pm.Gamma("yl", alpha=alpha[left_exact_idxs], beta=beta[left_exact_idxs], observed=left_exact)
    
    # Latent likelihood for inexact intervals (right)
    wr = pm.Potential('wr', censored('censoredr', alpha[right_interval_idxs], beta[right_interval_idxs], 
                                     right_interval[:, 1], right_interval[:, 0]))
    
    # Likelihood for exact intervals (right)
    yr = pm.Gamma("yr", alpha=alpha[right_exact_idxs], beta=beta[right_exact_idxs], observed=right_exact)

# Sample from the model
with mod:
    idata_dir = pm.sample(2000, tune=2000, nuts_sampler='numpyro', random_seed=27)


summ = az.summary(idata_dir, hdi_prob=0.9)
summ.to_csv("direct_analysis_summary.csv")

az.to_netcdf(idata_dir, "./direct_analysis_idata.nc")

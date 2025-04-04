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

def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)

# data = pd.read_csv("./data/data_h7n9_severity.csv")
# lower = data.IncP_min.values
# upper = data.IncP_max.values
# fatal = data.death_status.values   
# interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
# exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])

# read data
paths = glob.glob("./chunked_data/*")
paths = np.sort(paths)

dfs_sorted = [pd.read_csv(paths[i]) for i in range(len(paths))]

idx = np.arange(len(paths))

site_id = []
for i in idx:
    if i < 9: 
        s = "S0"+str(i+1)
    else:
        s = "S"+str(i+1)
    sid = np.repeat(s, len(dfs_sorted[i]))
    site_id.append(list(sid))
site_id = np.concatenate(site_id)


df_all = pd.concat(dfs_sorted)

df_all["site_id"] = site_id

n_obs = np.array([len(df) for df in dfs_sorted])

# Factorize SiteIdx to create integer indices
idxs = pd.factorize(df_all.site_id)[0]
sites_id = pd.unique(idxs)
n_sites = len(sites_id)

# Extract values from the DataFrame
lower = df_all.IncP_min.values
upper = df_all.IncP_max.values
fatal = df_all.death_status.values

interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal)])

# Separate exact and inexact intervals for left and right bounds
# Keep track of the original indices for alignment
interval_indices = np.array([i for i, (x,y) in enumerate(zip(lower, upper)) if x != y])
exact_indices = np.array([i for i, (x,y) in enumerate(zip(lower, upper)) if x == y])


# Extract data for exact and inexact intervals
interval = np.array([[lower[i], upper[i]] for i in interval_indices])
fatal_interval = np.array([[fatal[i]] for i in interval_indices])
exact = np.array([lower[i] for i in exact_indices])
fatal_exact = np.array([[fatal[i]] for i in exact_indices])

interval = np.array([[x,y,z] for x,y,z in zip(interval[:,0],interval[:,1],fatal_interval[:,0])])
exact = np.array([[x,y] for x,y in zip(exact, fatal_exact[:,0])])

# Extract the corresponding indices for exact and inexact data
interval_idxs = idxs[interval_indices]
exact_idxs = idxs[exact_indices]

with pm.Model() as mod:
    
    alpha_s = pm.HalfNormal("alpha_s", 0.5)
    beta_s = pm.HalfNormal("beta_s", 0.5)
    
    alpha_z = pm.Normal('alpha_z', 0, 0.1, shape=(2, n_sites)) 
    beta_z = pm.Normal('beta_z', 0, 0.1, shape=(2, n_sites))
    
    # Transformed parameters (ensuring positivity)
    alpha = pm.Deterministic("alpha", pt.exp(pt.log(3) + alpha_s * alpha_z)) # exp transform
    beta = pm.Deterministic("beta", pt.exp(pt.log(0.9) + beta_s * beta_z)) # exp transform
   
    mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
    
    # latent likelihood of 'inexact' incubation periods
    w = pm.Potential('w', censored('censored', alpha[interval[:,2], interval_idxs], 
                                               beta[interval[:,2], interval_idxs], 
                                               interval[:,0],
                                               interval[:,1]))
    
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    y = pm.Gamma("y", alpha[exact[:,1], exact_idxs], 
                      beta[exact[:,1], exact_idxs], 
                      observed=exact[:,0])


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

az.to_netcdf(idata_dir, "./direct_analysis_idata_sites.nc")


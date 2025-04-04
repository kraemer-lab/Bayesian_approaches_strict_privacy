# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pymc_experimental as pme
import pytensor.tensor as pt
make_prior = pme.utils.prior.prior_from_idata
from tqdm import tqdm
from scipy.stats import gamma

#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

np.random.seed(27) # set numpy random seed

# read data
paths = glob.glob("./chunked_data/*")
dfs_sorted = [pd.read_csv(paths[i]) for i in range(len(paths))]

n_sites = len(paths)

df_all = pd.concat(dfs_sorted)

df = dfs_sorted[0]

n_sites = len(paths)

# Factorize SiteIdx to create integer indices
idxs = df.index.values

n_obs = np.array([len(df) for df in dfs_sorted])

weights = pt.sqrt(1/n_obs)


# Extract values from the DataFrame
lower = df.min_incubation.values
upper = df.max_incubation.values

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


def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)

with pm.Model() as mod:
    
    alpha = pm.TruncatedNormal("alpha", mu=7, sigma=1, lower=0, upper=10)
    
    beta = pm.TruncatedNormal("beta", mu=0.9, sigma=0.2, lower=0, upper=2)
            
    mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
    
    # Latent likelihood for inexact intervals (left)
    w = pm.Potential('w', censored('censored', alpha, beta, 
                                     interval[:,0], interval[:,1]))
    
    y = pm.Gamma("y", alpha=alpha, beta=beta, observed=exact)

    ppc = pm.sample_prior_predictive(1000, random_seed=27)
    
mu_pri = az.extract(ppc.prior)['mu'].values
alp_pri = az.extract(ppc.prior)['alpha'].values
bet_pri = az.extract(ppc.prior)['beta'].values
plt.hist(mu_pri.T, bins=100)

with mod:
    idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                      random_seed=27)
try:
    del idata.observed_data
except:
    pass

del idata.sample_stats

idxs = np.arange(len(dfs_sorted))

idatas = [idata]

# names for saving idata by site and ordered number (idata01, idata02...etc)
names = []
for i in idxs:
    if i < 9:
        names.append("site0"+str(i+1)+"_idata.nc")
    else:
        names.append("site"+str(i+1)+"_idata.nc")

# save inference data from model
az.to_netcdf(idata, "./mvn_approx/"+names[0])

## loop over datasets, save and load idata to update prior
for i in tqdm(range(len(dfs_sorted[1:]))):
    if i+1 < 10:
        n = "0"+str(i+1)
    else:
        n = str(i+1)
    name = names[i]

    df = dfs_sorted[i]
    
    # Factorize SiteIdx to create integer indices
    idxs = df.index.values
    sites_id = pd.unique(idxs)
    
    # Extract values from the DataFrame
    lower = df.min_incubation.values
    upper = df.max_incubation.values
    
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
    
    pre_mu = az.extract(idatas[i].posterior)['mu'].values #mu from site s-1
    pre_mu_mu, pre_mu_sig = np.array([pre_mu.mean(), pre_mu.std()])
    
    pre_sigma = az.extract(idatas[i].posterior)['sigma'].values #sigma from site s-1
    pre_sig_mu, pre_sig_sig = np.array([pre_sigma.mean(), pre_sigma.std()]) 
    
    with pm.Model() as mod:
        
        prior = make_prior(idatas[i], var_names=['mu', 'sigma'])
        
        mu = prior['mu'] #* pre_mu_sig + pre_mu_mu
        sigma = prior['sigma'] #* pre_sig_sig + pre_sig_mu
    
        alpha = pm.Deterministic("alpha", mu**2 / sigma**2)
        
        beta = pm.Deterministic("beta", mu / sigma**2)

        if len(interval) > 0:
            # Latent likelihood for inexact intervals (left)
            w = pm.Potential('w', censored('censored', alpha, beta, 
                                             interval[:,0], interval[:,1]))
        
        if len(exact) > 0:
            # Likelihood for exact intervals (left)
            y = pm.Gamma("y", alpha, beta, observed=exact)

 
        idata = pm.sample(2000, tune=2000, chains=4, cores=12, random_seed=27,
                          target_accept=0.95, nuts_sampler="numpyro")
    try:
        del idata.observed_data
    except:
        pass
    del idata.sample_stats
    az.to_netcdf(idata, "./mvn_approx/"+name)
    idata = az.from_netcdf("./mvn_approx/"+name)
    idatas.append(idata)

try:
    del idatas[-1].observed_data
except:
    pass
try:
    del idatas[-1].sample_stats
except:
    pass
az.to_netcdf(idatas[-1], "./mvn_approx_idata.nc")



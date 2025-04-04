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

# Extract values from the DataFrame
left_lower = df.left_lower.values
left_upper = df.left_upper.values
right_lower = df.right_lower.values
right_upper = df.right_upper.values

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

idxs = np.arange(len(dfs_sorted))
sites_id = pd.unique(idxs)
n_sites = len(sites_id)

n_obs = np.array([len(df) for df in dfs_sorted])

weights = pt.sqrt(1/n_obs)

def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L + 1e-16) #add very small 1e-16 offset for stability

with pm.Model() as mod:
    
    alpha = pm.TruncatedNormal("alpha", mu=4, sigma=1, lower=0, upper=10)
    
    beta = pm.TruncatedNormal("beta", mu=0.66, sigma=0.2, lower=0, upper=2)

    mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
    
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
    
    # Latent likelihood for inexact intervals (left)
    wl = pm.Potential('wl', censored('censoredl', alpha, beta, 
                                     left_interval[:, 1], left_interval[:, 0]))
    
    # Likelihood for exact intervals (left)
    yl = pm.Gamma("yl", alpha=alpha, beta=beta, observed=left_exact)
    
    # Latent likelihood for inexact intervals (right)
    wr = pm.Potential('wr', censored('censoredr', alpha, beta, 
                                     right_interval[:, 1], right_interval[:, 0]))
    
    # Likelihood for exact intervals (right)
    yr = pm.Gamma("yr", alpha=alpha, beta=beta, observed=right_exact)

    
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

# # save inference data from model
# az.to_netcdf(idata, "./mvn_approx/"+names[0])

## loop over datasets, save and load idata to update prior
for i in tqdm(range(len(dfs_sorted[1:]))):
    if i+1 < 10:
        n = "0"+str(i+1)
    else:
        n = str(i+1)
    name = names[i]

    df = dfs_sorted[i]
    
    # Extract values from the DataFrame
    left_lower = df.left_lower.values
    left_upper = df.left_upper.values
    right_lower = df.right_lower.values
    right_upper = df.right_upper.values
    
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

    left_interval = np.array([[x,y] for x,y in zip(left_lower, right_lower)])
    
    right_interval = np.array([[x,y] for x,y in zip(left_upper, right_upper)])
    
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
                  
    # if len(left_exact) > 0:
        # Likelihood for exact intervals (left)
        yl = pm.Gamma("yl", alpha=alpha, beta=beta, observed=left_exact)
        
    # if len(left_interval) > 0:
        # Latent likelihood for inexact intervals (left)
        wl = pm.Potential('wl', censored('censoredl', alpha, beta, 
                                         left_interval[:, 1], left_interval[:, 0]))
              
    # if len(right_exact) > 0:
        # Likelihood for exact intervals (right)
        yr = pm.Gamma("yr", alpha=alpha, beta=beta, observed=right_exact)
        
    # if len(right_interval) > 0:
        # Latent likelihood for inexact intervals (right)
        wr = pm.Potential('wr', censored('censoredr', alpha, beta, 
                                         right_interval[:, 1], right_interval[:, 0]))
 
        idata = pm.sample(2000, tune=2000, chains=4, cores=12, random_seed=27) #target_accept=0.95)
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


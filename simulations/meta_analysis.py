# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pytensor.tensor as pt
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
paths = np.sort(paths)

dfs_sorted = [pd.read_csv(paths[i]) for i in range(len(paths))]

summs = glob.glob("./meta_analysis/*")
summs = np.sort(summs)


df_all = pd.concat(dfs_sorted)

def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)

idxs = np.arange(len(dfs_sorted))
sites_id = pd.unique(idxs)
n_sites = len(sites_id)

n_obs = np.array([len(df) for df in dfs_sorted])

weights = pt.sqrt(1/n_obs)

# names for saving idata by site and ordered number (idata01, idata02...etc)
names = []
for i in idxs:
    if i < 9:
        names.append("site0"+str(i+1)+"_idata_summary.csv")
    else:
        names.append("site"+str(i+1)+"_idata_summary.csv")
                     
if len(summs) < len(paths):
    summaries = []
    
    ## fit model to each lab
    for i in tqdm(range(len(dfs_sorted))):
        
        df = dfs_sorted[i]
        
        idxs = df.index.values
        
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
            
        with pm.Model() as mod:
            
            alpha = pm.TruncatedNormal("alpha", mu=7, sigma=1, lower=0, upper=10)
            
            beta = pm.TruncatedNormal("beta", mu=0.9, sigma=0.2, lower=0, upper=2)
    
            mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
            sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
            
            if len(interval) > 0:
                # Latent likelihood for inexact intervals (left)
                w = pm.Potential('w', censored('censored', alpha, beta, 
                                                 interval[:,0], interval[:,1]))
                
            if len(exact) > 0:
                # Likelihood for exact intervals (left)
                y = pm.Gamma("y", alpha=alpha, beta=beta, observed=exact)

            idata = pm.sample(2000, tune=2000, chains=4, cores=12, target_accept=0.95,
                              nuts_sampler="numpyro") 
                        
            summary = az.summary(idata, hdi_prob=0.9)
            summaries.append(summary)
            summary.to_csv("./meta_analysis/"+names[i])
            
            idata.close()    
        
else:
    pass


# # save inference data from model
# az.to_netcdf(idata, "./posterior_summary/"+names[0])

#looad sumamries
summs = glob.glob("./meta_analysis/*")
summaries = [pd.read_csv(s, index_col=0) for s in summs]

eff_sizes = np.array([s['mean']["mu"] for s in summaries])
errors =  np.array([s['sd']["mu"] for s in summaries])

with pm.Model() as meta_mod:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=5)  # Overall effect size
    tau = pm.HalfNormal('tau', 5)  # Between-study heterogeneity

    # True effect sizes for each study
    theta_z = pm.Normal('theta_z', mu=0, sigma=1, shape=len(eff_sizes))
    theta = pm.Deterministic("theta", mu + theta_z*tau)
    
    # Prior for true standard errors (sigma_i)
    sigma = pm.InverseGamma('sigma', mu=3, sigma=errors, shape=len(eff_sizes))
    
    alpha = pm.Deterministic("alpha", mu**2 / sigma**2)
    
    beta = pm.Deterministic("beta", mu / sigma**2)
            
    # Likelihood
    y = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=eff_sizes)

    # Sample from the posterior
    idata_meta = pm.sample(2000, tune=2000, nuts_sampler='numpyro', target_accept=0.99,
                      random_seed=27)

try:
    del idata_meta.observed_data
except:
    pass
try:
    del idata_meta.sample_stats
except:
    pass
az.to_netcdf(idata_meta, "./meta_analysis_idata.nc")

summ = az.summary(idata_meta, hdi_prob=0.9)
summ.to_csv("meta_analysis_summary.csv")



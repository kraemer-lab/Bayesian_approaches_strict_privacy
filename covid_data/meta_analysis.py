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

sampled_summaries_path = glob.glob("./meta_analysis/*")
# summs = np.sort(summs)

#define how many dfs are left to sample
to_sample = np.arange(len(paths) - len(sampled_summaries_path)) + len(sampled_summaries_path)

df_all = pd.concat(dfs_sorted)

## ICG likelihood function
def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)

idxs = np.arange(len(dfs_sorted))
sites_id = pd.unique(idxs)
n_sites = len(sites_id)

n_obs = np.array([len(df) for df in dfs_sorted])


# names for saving idata by site and ordered number (idata01, idata02...etc)
names = []
for i in idxs:
    if i < 9:
        names.append("site0"+str(i+1)+"_idata_summary.csv")
    else:
        names.append("site"+str(i+1)+"_idata_summary.csv")


summaries = [pd.read_csv(summ) for summ in sampled_summaries_path]

## fit model to each lab
for i in tqdm(to_sample):
    
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
        
    with pm.Model() as mod:
        
        alpha = pm.TruncatedNormal("alpha", mu=4, sigma=1, lower=0, upper=10)
        
        beta = pm.TruncatedNormal("beta", mu=0.66, sigma=0.2, lower=0, upper=2)
    
        mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
        
        sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
        
    # if len(left_exact) > 0:
        # Likelihood for exact intervals (left)
        yl = pm.Gamma("yl", mu=mu, sigma=sigma, observed=left_exact)
    # if len(left_interval) > 0:
        # Latent likelihood for inexact intervals (left)
        wl = pm.Potential('wl', censored('censoredl', alpha, beta, 
                                         left_interval[:, 1], left_interval[:, 0]))
              
    # if len(right_exact) > 0:
        # Likelihood for exact intervals (right)
        yr = pm.Gamma("yr", mu=mu, sigma=sigma, observed=right_exact)
    # if len(right_interval) > 0:
        # Latent likelihood for inexact intervals (right)
        wr = pm.Potential('wr', censored('censoredr', alpha, beta, 
                                         right_interval[:, 1], right_interval[:, 0]))

   
        idata = pm.sample(2000, tune=2000, chains=4, cores=12, target_accept=0.99,
                          nuts_sampler="numpyro") 
                    
        summary = az.summary(idata)
        summaries.append(summary)
        summary.to_csv("./meta_analysis/"+names[i])
        
        idata.close()    
        
else:
    pass



#load sumamries
summs = glob.glob("./meta_analysis/*")
summaries = [pd.read_csv(s, index_col=0) for s in summs]

eff_sizes = np.array([s['mean']["mu"] for s in summaries])
errors =  np.array([s['sd']["mu"] for s in summaries])

with pm.Model() as meta_mod:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=1)  # Overall effect size
    tau = pm.HalfNormal('tau', 1)  # Between-study heterogeneity

    # True effect sizes for each study
    theta_z = pm.Normal('theta_z', mu=0, sigma=1, shape=len(eff_sizes))
    theta = pm.Deterministic("theta", mu + theta_z*tau)
    
    # Prior for true standard errors (sigma_i)
    sigma = pm.InverseGamma('sigma', mu=3, sigma=errors, shape=len(eff_sizes))
            
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

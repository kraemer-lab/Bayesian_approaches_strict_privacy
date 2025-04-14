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

#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

np.random.seed(27) # set numpy random seed

# read data
paths = glob.glob("./chunked_data/*")
dfs = [pd.read_csv(paths[i]) for i in range(len(paths))]
dfs_sorted = dfs

df_all = pd.concat(dfs)

n_sites = len(paths)

idxs = np.arange(n_sites)

## ICG likelihood function
def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)


summs = []

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
        
        df = dfs[i]
        
        # Extract values from the DataFrame
        lower = df.IncP_min.values
        upper = df.IncP_max.values
        fatal = df.death_status.values
        
        lower = df.IncP_min.values #incubation periods lower boundary
        upper = df.IncP_max.values #incubation periods upper boundary
    
        fatal = df.death_status.values
    
        interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
        exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])
        
        with pm.Model() as mod:
            
            alpha = pm.TruncatedNormal('alpha', 3, 1, lower=0, upper=10, shape=2)
            
            beta = pm.TruncatedNormal('beta', 0.9, 0.2, lower=0, upper=5, shape=2)
    
            mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
            
            sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
            
            # latent likelihood of 'inexact' incubation periods
            w = pm.Potential('w', censored('censored', alpha[interval[:,2]], 
                                                       beta[interval[:,2]], 
                                                       interval[:,0],
                                                       interval[:,1]))
            
            # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
            y = pm.Gamma("y", alpha[exact[:,1]], beta[exact[:,1]], observed=exact[:,0])
            
            idata = pm.sample(2000, tune=2000, chains=4, cores=12, 
                              random_seed=27, target_accept=0.95, nuts_sampler="numpyro")

    
        summary = az.summary(idata)
        summaries.append(summary)
        summary.to_csv("./meta_analysis/"+names[i])
else:
    pass



#loop over datasets to perform meta-analysis
summs = glob.glob("./meta_analysis/*")
summaries = [pd.read_csv(s, index_col=0) for s in summs]

eff_sizes1 = np.array([s['mean']["mu[0]"] for s in summaries])
errors1 =  np.array([s['sd']["mu[0]"] for s in summaries])
eff_sizes2 = np.array([s['mean']["mu[1]"] for s in summaries])
errors2 =  np.array([s['sd']["mu[1]"] for s in summaries])

eff_sizes = np.array([eff_sizes1, eff_sizes2])
errors = np.array([errors1, errors2])


with pm.Model() as meta_mod:
    
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=5, shape=2)  # Overall effect size
    tau = pm.HalfNormal('tau', 5, shape=2)  # Between-study heterogeneity

    # True effect sizes for each study
    theta_z = pm.Normal('theta_z', mu=0, sigma=1, shape=eff_sizes.shape)
    theta = pm.Deterministic("theta", (mu + theta_z.T*tau).T)
    
    # Prior for true standard errors (sigma_i)
    # sigma = pm.InverseGamma('sigma', alpha=2, beta=errors, shape=eff_sizes.shape)
    #sigma = pm.TruncatedNormal('sigma', mu=2, sigma=errors, shape=eff_sizes.shape, lower=0)
    sigma = pm.InverseGamma("sigma", mu=2, sigma=errors, shape=eff_sizes.shape)
    
    alpha = pm.Deterministic("alpha", mu**2 / sigma.mean(axis=1)**2)
    beta = pm.Deterministic("beta", mu / sigma.mean(axis=1)**2)
    
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


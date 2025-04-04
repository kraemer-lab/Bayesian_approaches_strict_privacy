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

data = pd.read_csv("./data/data_h7n9_severity.csv")
lower = data.IncP_min.values
upper = data.IncP_max.values
fatal = data.death_status.values   
interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])


with pm.Model() as mod:
    
    alpha_s = pm.HalfNormal("alpha_s", 0.5)
    beta_s = pm.HalfNormal("beta_s", 0.5)
    
    alpha_z = pm.Normal('alpha_z', 0, 0.1, shape=2) 
    beta_z = pm.Normal('beta_z', 0, 0.1, shape=2)
    
    # Transformed parameters (ensuring positivity)
    alpha = pm.Deterministic("alpha", pt.exp(pt.log(3) + alpha_s * alpha_z)) # exp transform
    beta = pm.Deterministic("beta", pt.exp(pt.log(0.9) + beta_s * beta_z)) # exp transform
   
    mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
    
    # latent likelihood of 'inexact' incubation periods
    w = pm.Potential('w', censored('censored', alpha[interval[:,2]], 
                                               beta[interval[:,2]], 
                                               interval[:,0],
                                               interval[:,1]))
    
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    y = pm.Gamma("y", alpha[exact[:,1]], 
                      beta[exact[:,1]], 
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

az.to_netcdf(idata_dir, "./direct_analysis_idata.nc")

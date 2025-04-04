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

df_all = pd.concat(dfs)

n_sites = len(paths)

# order datasets from largest no. of datapoints to smallest
len_dfs = [len(d) for d in dfs]
len_dfs2 = sorted(len_dfs.copy(), reverse=True)
idxs = []
for d in range(len(dfs)):
    idx = np.where(np.array(len_dfs)==len_dfs2[d])[0][0]
    idxs.append(idx)
dfs_sorted = [dfs[i] for i in idxs] 

def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)

data = dfs_sorted[0]

lower = data.IncP_min.values #incubation periods lower boundary
upper = data.IncP_max.values #incubation periods upper boundary

fatal = data.death_status.values

interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])

# Sample the initial model from largest dataset
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
    
    ppc = pm.sample_prior_predictive(1000, random_seed=27)

mu_pri = az.extract(ppc.prior)['mu'].values
alp_pri = az.extract(ppc.prior)['alpha'].values
bet_pri = az.extract(ppc.prior)['beta'].values
plt.hist(mu_pri.T, bins=100)

with mod:
    idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                      random_seed=27)

del idata.observed_data

idatas = [idata]

# names for saving idata by site and ordered number (idata01, idata02...etc)
names = ["site"+str(idxs[0]+1)+"_idata01.nc"]

# save inference data from model
az.to_netcdf(idata, "./mvn_approx/"+names[0])

## loop over datasets, save and load idata to update prior

for i in tqdm(range(len(dfs_sorted[1:]))):
    if i+1 < 10:
        n = "0"+str(i+1)
    else:
        n = str(i+1)
    
    name = "site"+str(idxs[i]+1)+"_idata"+n+".nc"
    names.append(name)
    
    df = dfs_sorted[1:][i]
    
    lower = df.IncP_min.values #incubation periods lower boundary
    upper = df.IncP_max.values #incubation periods upper boundary

    fatal = df.death_status.values

    interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
    exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])
    
    pre_mu = az.extract(idatas[i].posterior)['mu'].values #mu from site s-1
    pre_mu_mu1, pre_mu_sig1 = np.array([pre_mu[0].mean(), pre_mu[0].std()]) #sp.stats.norm.fit(pre_mu[0])
    pre_mu_mu2, pre_mu_sig2 = np.array([pre_mu[1].mean(), pre_mu[1].std()]) #sp.stats.norm.fit(pre_mu[1])
    pre_sigma = az.extract(idatas[i].posterior)['sigma'].values #sigma from site s-1
    pre_sig_mu1, pre_sig_sig1 = np.array([pre_sigma[0].mean(), pre_sigma[0].std()]) #sp.stats.norm.fit(pre_sigma[0])
    pre_sig_mu2, pre_sig_sig2 = np.array([pre_sigma[1].mean(), pre_sigma[1].std()]) #sp.stats.norm.fit(pre_sigma[1])
   
    with pm.Model() as mod:
        prior = make_prior(idatas[i], var_names=['mu', 'sigma'])
        # alpha = prior['alpha']
        # beta = prior['beta']
        # mu = pm.Deterministic('mu', alpha/beta)
        # sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2))
        
        ms = pt.as_tensor_variable([pre_mu_sig1, pre_mu_sig2])
        mm = pt.as_tensor_variable([pre_mu_mu1, pre_mu_mu2])
        ss = pt.as_tensor_variable([pre_sig_sig1, pre_sig_sig2])
        sm = pt.as_tensor_variable([pre_sig_mu1, pre_sig_mu2])
        
        mu = prior['mu'] #* ms + mm
        sigma = prior['sigma'] #* ss + sm 
        
        alpha = pm.Deterministic("alpha", mu**2 / sigma**2)
        beta = pm.Deterministic("beta", mu / sigma**2)
        
        w = pm.Potential('w', censored('censored', mu[interval[:,2]], 
                                                   sigma[interval[:,2]], 
                                                   interval[:,0],
                                                   interval[:,1]))
        
        # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
        y = pm.Gamma("y", mu=mu[exact[:,1]], sigma=sigma[exact[:,1]], observed=exact[:,0])
        
        idata = pm.sample(2000, tune=2000, chains=4, cores=12, 
                          random_seed=27, nuts_sampler="numpyro")
    
    del idata.observed_data #remove observed data from idata for safety
    
    az.to_netcdf(idata, "./mvn_approx/"+names[1:][i])
    idata = az.from_netcdf("./mvn_approx/"+names[1:][i])
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


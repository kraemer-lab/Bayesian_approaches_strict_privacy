# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt

np.random.seed(27)

data = pd.read_csv("./data/data_h7n9_severity.csv")

df = data.sample(frac=1).reset_index(drop=True) #shuffle data

chunks = [18, 20, 26, 31, 42, 51, 55, 46, 35, 28, 24, 19]

for c in range(len(chunks)):
    if c < 10:
        site = "S0"+str(c+1)
    else:
        site = "S"+str(c+1)
    d = df[:chunks[c]]
    d.to_csv("chunked_data/df_"+str(site)+".csv", index=False)
    idx = d.index
    df = df.drop(idx)


#####################################
########## Weibull model ############

def weibull_cdf(x, kappa, theta):
    return 1 - pt.exp(-(x/theta)**kappa)

def censored(name, kappa, theta, lower, upper):
    L = weibull_cdf(lower, kappa, theta)
    U = weibull_cdf(upper, kappa, theta)
    return pt.log(U - L)

data = pd.read_csv("./data/data_h7n9_severity.csv")

lower = data.IncP_min.values
upper = data.IncP_max.values

fatal = data.death_status.values   

lower = data.IncP_min.values #incubation periods lower boundary
upper = data.IncP_max.values #incubation periods upper boundary

fatal = data.death_status.values

interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])

with pm.Model() as mod:
    kappa = pm.Uniform('kappa', 0, 10, shape=2) #as used in the paper
    theta = pm.Uniform('theta', 0, 10, shape=2) #as used in the paper
    mu = pm.Deterministic('mu', theta*pt.gamma(1 + 1 / kappa)) #incubation period mean
    # latent likelihood of 'inexact' incubation periods
    w = pm.Potential('w', censored('censored', kappa[interval[:,2]], 
                                               theta[interval[:,2]], 
                                               interval[:,0],
                                               interval[:,1]))
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    y = pm.Weibull("y", kappa[exact[:,1]], theta[exact[:,1]], observed=exact[:,0])


with mod:
    idata = pm.sample(1000, tune=1000, nuts_sampler='numpyro', random_seed=27)

az.plot_trace(idata)

summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("weibull_summary_direct.csv")


#####################################
########## Gamma model ############
def gamma_cdf(x, alpha, beta):
    return pt.gammainc(alpha, x*beta) #/ pt.gamma(alpha) 

def censored(name, alpha, beta, lower, upper):
    L = gamma_cdf(lower, alpha, beta)
    U = gamma_cdf(upper, alpha, beta)
    return pt.log(U - L)

fatal = data.death_status.values

interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])

with pm.Model() as mod:
    alpha = pm.TruncatedNormal('alpha', 1, 10, lower=1, upper=30, shape=2) 
    beta = pm.TruncatedNormal('beta', 1, 10, lower=1, upper=2, shape=2)
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
    idata = pm.sample(1000, tune=1000, nuts_sampler='numpyro', random_seed=27)


az.plot_trace(idata)

summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("gamma_summary_direct.csv")



#####################################
########## NB model ############
def gamma_cdf(x, alpha, beta):
    return pt.gammainc(alpha, x*beta) #/ pt.gamma(alpha) 

def censored(name, alpha, beta, lower, upper):
    L = gamma_cdf(lower, alpha, beta)
    U = gamma_cdf(upper, alpha, beta)
    return pt.log(U - L)

fatal = data.death_status.values

interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])

with pm.Model() as mod:
    alpha = pm.TruncatedNormal('alpha', 1, 10, lower=1, upper=30, shape=2) 
    beta = pm.TruncatedNormal('beta', 1, 10, lower=1, upper=2, shape=2)
    mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
    # latent likelihood of 'inexact' incubation periods
    w = pm.Potential('w', censored('censored', alpha[interval[:,2]], 
                                               beta[interval[:,2]], 
                                               interval[:,0],
                                               interval[:,1]))
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    y = pm.NegativeBinomial("y", mu=mu[exact[:,1]], alpha=alpha[exact[:,1]], observed=exact[:,0])


with mod:
    idata = pm.sample(2000, tune=2000, nuts_sampler='numpyro', random_seed=27)


az.plot_trace(idata)

summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("nb_summary_direct.csv")
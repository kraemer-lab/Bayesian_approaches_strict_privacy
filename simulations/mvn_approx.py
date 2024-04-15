# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import glob
from scipy.optimize import minimize
import itertools
import pymc_experimental as pme
make_prior = pme.utils.prior.prior_from_idata

#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

np.random.seed(27) # set numpy random seed

# read data
paths = glob.glob("./sim_data/*")
dfs = [pd.read_csv(paths[i]) for i in range(len(paths))]

df_all = pd.concat(dfs)

# order datasets from largest no. of datapoints to smallest
len_dfs = [len(d) for d in dfs]
len_dfs2 = sorted(len_dfs.copy(), reverse=True)
idxs = []
for d in range(len(dfs)):
    idx = np.where(np.array(len_dfs)==len_dfs2[d])[0][0]
    idxs.append(idx)
dfs_sorted = [dfs[i] for i in idxs] 


# Sample the initial model from largest dataset
with pm.Model() as mod0:
    lamz = pm.Normal('lamz', mu=0, sigma=2)
    lam = pm.Normal('lam', mu=0, sigma=2)
    sigma = pm.TruncatedNormal('sigma', mu=0, sigma=2, lower=0)
    mu = pm.Deterministic("mu", pm.math.exp(lam + lamz*sigma))
    alpha = pm.TruncatedNormal('alpha', mu=0, sigma=2, lower=0)
    y = pm.NegativeBinomial('y', mu=mu, alpha=alpha, observed=dfs_sorted[0].days.values)
    idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                      target_accept=0.95, random_seed=27)

del idata.observed_data

idatas = [idata]

# names for saving idata by site and ordered number (idata01, idata02...etc)
names = ["site"+str(idxs[0]+1)+"_idata01.nc"]

# save inference data from model
az.to_netcdf(idata, "./mvn_approx/"+names[0])

#loop over datasets, save and load idata to update prior
for i in range(len(dfs_sorted[1:])):
    if i+1 < 10:
        n = "0"+str(i+1)
    else:
        n = str(i+1)
    name = "site"+str(idxs[i]+1)+"_idata"+n+".nc"
    names.append(name)
    df = dfs_sorted[1:][i]
    with pm.Model() as mod:
        prior = make_prior(idatas[i], var_names=['lamz', 'lam', 'sigma', 'alpha'])
        lamz = prior['lamz']
        lam = prior['lam']
        sigma = prior['sigma']
        alpha = prior['alpha']
        mu = pm.Deterministic("mu", pm.math.exp(lam + lamz*sigma))
        y = pm.NegativeBinomial('y', mu=mu, alpha=alpha, observed=df.days.values)
        idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                          target_accept=0.95, random_seed=27)
    del idata.observed_data
    #az.to_netcdf(idata, "./mvn_approx/"+names[1:][i])
    #idata = az.from_netcdf("./mvn_approx/"+names[1:][i])
    idatas.append(idata)

##save summary of latest sampled site
summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("mvn_approx_summary.csv")


## Plot CDF
pos_mu = az.extract(idata)['mu'].values
pos_alp = az.extract(idata)['alpha'].values
pos_var = pos_mu + pos_mu**2 / pos_alp
pos_sd = np.sqrt(pos_var)

pos_up = pos_mu+pos_sd
var_up = pos_up + pos_up**2 / pos_alp
pos_low = pos_mu-pos_sd
var_low = pos_low + pos_low**2 / pos_alp

pos_p = pos_mu / pos_var
pos_n = (pos_mu**2) / (pos_var - pos_mu)

p_low = pos_low / var_low
n_low = (pos_low**2) / (var_low - pos_low)
p_up = pos_up / var_up
n_up = (pos_up**2) / (var_up - pos_up)

x = np.array([np.arange(30, step=0.1) for i in range(pos_mu.shape[0])]).T #30 days

pos_cdf = sp.stats.nbinom.cdf(x, pos_n, pos_p)

h5, h95 = az.hdi(pos_cdf.T, hdi_prob=0.9).T

s_low = sp.stats.nbinom.cdf(x, pos_n, p_low).mean(axis=1)
s_up = sp.stats.nbinom.cdf(x, pos_n, p_up).mean(axis=1)


cdf_mean = pos_cdf.mean(axis=1)
x2 = np.arange(30, step=0.1)

fig, ax = plt.subplots()
sns.ecdfplot(df_all.days.values, ax=ax, color='k',  label="Observed")
ax.plot(x2, cdf_mean, color='orangered', linestyle="--", label="μ mean")
ax.fill_between(x2, s_low, s_up, color='orangered', alpha=0.2, label="±SD mean")
plt.legend()
ax.set_axisbelow(True)
ax.grid(alpha=0.2)
ax.set_ylabel("Probability")
ax.set_xlabel("Incubation Days")
ax.set_title("MvN Approximation Posterior CDF")
plt.tight_layout()
plt.savefig("mvn_approx_posterior.png", dpi=600)
plt.show()
plt.close()

# pos_mu = az.extract(idata)['mu'].values

# print("μ posterior mean: "+str(pos_mu.mean().round(2)))

# h5, h95 = az.hdi(pos_mu, hdi_prob=0.9)
# pos_alp =  az.extract(idata)['alpha'].values
# pos_sd = np.sqrt((pos_mu**2)/pos_alp + pos_mu)
# x = np.array([np.arange(30, step=0.1) for i in range(pos_mu.shape[0])]).T #30 days
# mean_cdfs = sp.stats.poisson.cdf(x, pos_mu)
# h5, h95 = az.hdi(mean_cdfs.T, hdi_prob=0.9).T
# mme = mean_cdfs.mean(axis=1)
# alp_cdfs = sp.stats.poisson.cdf(x, pos_alp)
# s_low = sp.stats.poisson.cdf(x, pos_mu-pos_sd).mean(axis=1)
# s_up = sp.stats.poisson.cdf(x, pos_mu+pos_sd).mean(axis=1)
# x2 = np.arange(30, step=0.1)

# fig, ax = plt.subplots()
# sns.ecdfplot(df_all.days.values, ax=ax, color='k',  label="Observed")
# ax.plot(x2, mme, color='orangered', linestyle="--", label="μ mean")
# ax.fill_between(x2, s_low, s_up, color='orangered', alpha=0.2, label="±SD mean")
# plt.legend()
# ax.set_axisbelow(True)
# ax.grid(alpha=0.2)
# ax.set_ylabel("Probability")
# ax.set_xlabel("Incubation Days")
# ax.set_title("MvN Approach Posterior CDF")
# plt.tight_layout()
# plt.savefig("mvn_approx_posterior.png", dpi=600)
# plt.show()
# plt.close()


# Fit model to all data.
site_idx = df_all.site.values
sites = len(df_all.site.unique())

with pm.Model() as mod:
    lamz = pm.Normal('lamz', mu=0, sigma=2, shape=sites)
    lam = pm.Normal('lam', mu=0, sigma=2)
    sigma = pm.TruncatedNormal('sigma', mu=0, sigma=2, lower=0)
    mu = pm.Deterministic("mu", pm.math.exp(lam + lamz*sigma))
    alpha = pm.TruncatedNormal('alpha', mu=0, sigma=2, lower=0)
    y = pm.NegativeBinomial('y', mu=mu[site_idx], alpha=alpha, observed=df_all.days.values)
    
with mod:
    idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                      target_accept=0.95, random_seed=27)


pos_mu_all = az.extract(idata)['mu'].values.mean(axis=0)
pos_alp_all = az.extract(idata)['alpha'].values.mean(axis=0)
pos_var = pos_mu + pos_mu**2 / pos_alp
pos_sd = np.sqrt(pos_var)
pos_p = pos_mu / pos_var
pos_n = (pos_mu**2) / (pos_var - pos_mu)
pos_cdf_all = sp.stats.nbinom.cdf(x, pos_n, pos_p)
cdf_all_mean = pos_cdf_all.mean(axis=1)

sqrt_part = np.sqrt((2*pos_mu_all.std() * pos_mu.std()) / (pos_mu_all.std()**2 + pos_mu.std()**2))
exp_part = np.exp(-0.25 * (pos_mu_all.mean() - pos_mu.mean())**2 / (pos_mu_all.std()**2 + pos_mu.std()**2))

H2 = np.round(1 - sqrt_part * exp_part, 3)

## Plot comparison between direct sampling and posterior update
fig, ax = plt.subplots(1,2, figsize=(12,6))
sns.kdeplot(pos_mu_all, color='forestgreen', linewidth=3, ax=ax[0])
sns.kdeplot(pos_mu, color='purple', linestyle='--', linewidth=3, ax=ax[0])
ax[0].text(10, 1.75, "H² = "+str(H2))
ax[0].set_axisbelow(True)
ax[0].grid(alpha=0.2)
ax[0].set_ylabel("Density")
ax[0].set_xlabel("Incubation Days")
ax[0].set_title("Posterior Distributions (μ)")
ax[1].plot(x2, cdf_all_mean, color='forestgreen', linewidth=3, label="μ mean Direct Sampling")
ax[1].plot(x2, cdf_mean, color='purple', linestyle="--", linewidth=3, label="μ mean MvN Approx.")
ax[1].legend()
ax[1].set_axisbelow(True)
ax[1].grid(alpha=0.2)
ax[1].set_ylabel("Probability")
ax[1].set_xlabel("Incubation Days")
ax[1].set_title("Posterior Distributions (μ) CDFs")
plt.suptitle("MvN Approach vs Direct Sampling")
plt.tight_layout()
plt.savefig("posterior_mvn_comparison.png", dpi=600)
plt.show()
plt.close()
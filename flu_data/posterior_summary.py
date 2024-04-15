# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import glob
import pymc_experimental as pme
import pytensor.tensor as pt
make_prior = pme.utils.prior.prior_from_idata

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

# order datasets from largest no. of datapoints to smallest
len_dfs = [len(d) for d in dfs]
len_dfs2 = sorted(len_dfs.copy(), reverse=True)
idxs = []
for d in range(len(dfs)):
    idx = np.where(np.array(len_dfs)==len_dfs2[d])[0][0]
    idxs.append(idx)
dfs_sorted = [dfs[i] for i in idxs] 

def gamma_cdf(x, alpha, beta):
    return pt.gammainc(alpha, x*beta) #/ pt.gamma(alpha) 

def censored(name, alpha, beta, lower, upper):
    L = gamma_cdf(lower, alpha, beta)
    U = gamma_cdf(upper, alpha, beta)
    return pt.log(U - L)

data = dfs_sorted[0]

lower = data.IncP_min.values #incubation periods lower boundary
upper = data.IncP_max.values #incubation periods upper boundary

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
    idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                      random_seed=27)

del idata.observed_data

summary = az.summary(idata)

summaries = [summary]

# names for saving idata by site and ordered number (idata01, idata02...etc)
names = ["site"+str(idxs[0]+1)+"_idata01.csv"]

# # save inference data from model
# az.to_netcdf(idata, "./posterior_summary/"+names[0])

#loop over datasets, save and load idata to update prior
def norm_cdf(x, mu, sigma):
    return 0.5 * (1 + pt.erf( (x-mu) / (sigma*pt.sqrt(2)) )) 

def censored(name, mu, sigma, lower, upper):
    L = norm_cdf(lower, mu, sigma)
    U = norm_cdf(upper, mu, sigma)
    return pt.log(U - L)

for i in range(len(dfs_sorted[1:])):
    mu_m0 = summaries[i].loc['mu[0]']['mean']
    mu_sd0 = summaries[i].loc['mu[0]']['sd']
    sigma_m0 = summaries[i].loc['sigma[0]']['mean']
    sigma_sd0 = summaries[i].loc['sigma[0]']['sd']
    mu_m1 = summaries[i].loc['mu[1]']['mean']
    mu_sd1 = summaries[i].loc['mu[1]']['sd']
    sigma_m1 = summaries[i].loc['sigma[1]']['mean']
    sigma_sd1 = summaries[i].loc['sigma[1]']['sd']
    if i+1 < 10:
        n = "0"+str(i+1)
    else:
        n = str(i+1)
    name = "site"+str(idxs[i]+1)+"_idata"+n+".csv"
    names.append(name)
    df = dfs_sorted[1:][i]
    lower = df.IncP_min.values #incubation periods lower boundary
    upper = df.IncP_max.values #incubation periods upper boundary
    fatal = df.death_status.values
    interval = np.array([[x,y,f] for x,y,f in zip(lower,upper,fatal) if x!=y])
    exact = np.array([[x,f] for x,y,f in zip(lower, upper, fatal) if x==y])
    with pm.Model() as mod:
        mu0 = pm.TruncatedNormal('mu0', mu_m0, mu_sd0, lower=1) 
        sigma0 = pm.TruncatedNormal('sigma0', sigma_m0, sigma_sd0, lower=0) 
        mu1 = pm.TruncatedNormal('mu1', mu_m1, mu_sd1, lower=1) 
        sigma1 = pm.TruncatedNormal('sigma1', sigma_m1, sigma_sd1, lower=0)
        mu = pm.Deterministic('mu', pt.as_tensor_variable([mu0,mu1]))
        sigma = pm.Deterministic('sigma', pt.as_tensor_variable([sigma0,sigma1]))
        # latent likelihood of 'inexact' incubation periods
        w = pm.Potential('w', censored('censored', mu[interval[:,2]], 
                                                   sigma[interval[:,2]], 
                                                   interval[:,0],
                                                   interval[:,1]))
        # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
        y = pm.Gamma("y", mu=mu[exact[:,1]], sigma=sigma[exact[:,1]], observed=exact[:,0])
        idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                          random_seed=27)
    del idata.observed_data
    summary = az.summary(idata)
    summaries.append(summary)
    summary.to_csv("./posterior_summary/"+names[i])

summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("posterior_summary_summary.csv")

pos_mu_1 = az.extract(idata)['mu'].values[0]
pos_mu_2 = az.extract(idata)['mu'].values[1]
pos_sig_1 = az.extract(idata)['sigma'].values[0]
pos_sig_2 = az.extract(idata)['sigma'].values[1]


### Sample all data directly
def gamma_cdf(x, alpha, beta):
    return pt.gammainc(alpha, x*beta) #/ pt.gamma(alpha) 

def censored(name, alpha, beta, lower, upper):
    L = gamma_cdf(lower, alpha, beta)
    U = gamma_cdf(upper, alpha, beta)
    return pt.log(U - L)

data = pd.read_csv("./data/data_h7n9_severity.csv")
lower = data.IncP_min.values
upper = data.IncP_max.values
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


with mod:
    idata = pm.sample(2000, tune=2000, nuts_sampler='numpyro', random_seed=27)

pos_mu_all_1 = az.extract(idata)['mu'].values[0]
pos_mu_all_2 = az.extract(idata)['mu'].values[1]
pos_sig_all_1 = az.extract(idata)['sigma'].values[0]
pos_sig_all_2 = az.extract(idata)['sigma'].values[1]

####compute hellinger indices
sqrt_part = np.sqrt((2*pos_mu_all_1.std() * pos_mu_1.std()) / (pos_mu_all_1.std()**2 + 
                                                               pos_mu_1.std()**2))
exp_part = np.exp(-0.25 * (pos_mu_all_1.mean() - pos_mu_1.mean())**2 / (pos_mu_all_1.std()**2 + 
                                                                        pos_mu_1.std()**2))
H2_1 = np.round(1 - sqrt_part * exp_part, 2)

sqrt_part = np.sqrt((2*pos_mu_all_2.std() * pos_mu_2.std()) / (pos_mu_all_2.std()**2 + 
                                                               pos_mu_2.std()**2))
exp_part = np.exp(-0.25 * (pos_mu_all_2.mean() - pos_mu_2.mean())**2 / (pos_mu_all_2.std()**2 + 
                                                                        pos_mu_2.std()**2))
H2_2 = np.round(1 - sqrt_part * exp_part, 2)



x = np.array([np.arange(30, step=0.1) for i in range(pos_mu_1.shape[0])]).T #30 days


cdf_1 = sp.stats.gamma.cdf(x, pos_mu_1**2 / pos_sig_1**2) 
cdf_2 = sp.stats.gamma.cdf(x, pos_mu_2**2 / pos_sig_2**2) 

cdf_all_1 = sp.stats.gamma.cdf(x, pos_mu_all_1**2 / pos_sig_all_1**2) 
cdf_all_2 = sp.stats.gamma.cdf(x, pos_mu_all_2**2 / pos_sig_all_2**2) 

### compute Gamma pdfs
pdf_1 = sp.stats.gamma.pdf(x, pos_mu_1**2 / pos_sig_1**2) 
pdf_2 = sp.stats.gamma.pdf(x, pos_mu_2**2 / pos_sig_2**2) 

pdf_all_1 = sp.stats.gamma.pdf(x, pos_mu_all_1**2 / pos_sig_all_1**2) 
pdf_all_2 = sp.stats.gamma.pdf(x, pos_mu_all_2**2 / pos_sig_all_2**2) 


## Plot comparison between direct sampling and posterior update
x2 = np.arange(30, step=0.1)
fig, ax = plt.subplots(2,2, figsize=(12,8))
sns.kdeplot(pos_mu_all_1, color='forestgreen', linestyle='-.', linewidth=3, ax=ax[1,0])
sns.kdeplot(pos_mu_1, color='purple', linestyle=':', linewidth=3, ax=ax[1,0])
sns.kdeplot(pos_mu_all_2, color='dodgerblue', linewidth=3, ax=ax[1,0])
sns.kdeplot(pos_mu_2, color='orangered', linestyle='--', linewidth=3, ax=ax[1,0])
ax[1,0].text(4.1, 1.2, "G1 H² = "+str(H2_1))
ax[1,0].text(4.1, 1, "G2 H² = "+str(H2_2))
ax[1,0].set_axisbelow(True)
ax[1,0].grid(alpha=0.2)
ax[1,0].set_ylabel("Density")
ax[1,0].set_xlabel("Incubation Days")
ax[1,0].set_title("C. Posterior Distributions (μ)")
ax[0,1].plot(x2, cdf_all_1.mean(axis=1), color='forestgreen', linestyle='-.', linewidth=3, label="G1 Direct Sampling")
ax[0,1].plot(x2, cdf_1.mean(axis=1), color='purple', linestyle=":", linewidth=3, label="G1 TN Approx.")
ax[0,1].plot(x2, cdf_all_2.mean(axis=1), color='dodgerblue', linewidth=3, label="G2 Direct Sampling")
ax[0,1].plot(x2, cdf_2.mean(axis=1), color='orangered', linestyle="--", linewidth=3, label="G2 TN Approx.")
ax[0,1].legend()
ax[0,1].set_axisbelow(True)
ax[0,1].grid(alpha=0.2)
ax[0,1].set_ylabel("Probability")
ax[0,1].set_xlabel("Incubation Days")
ax[0,1].set_title("B. Gamma CDFs")
ax[0,0].plot(x2, pdf_all_1.mean(axis=1), color='forestgreen', linestyle='-.', linewidth=3, label="G1 Direct Sampling")
ax[0,0].plot(x2, pdf_1.mean(axis=1), color='purple', linestyle=":", linewidth=3, label="G1 TN Approx.")
ax[0,0].plot(x2, pdf_all_2.mean(axis=1), color='dodgerblue', linewidth=3, label="G2 Direct Sampling")
ax[0,0].plot(x2, pdf_2.mean(axis=1), color='orangered', linestyle="--", linewidth=3, label="G2 TN Approx.")
ax[0,0].set_axisbelow(True)
ax[0,0].grid(alpha=0.2)
ax[0,0].set_ylabel("Density")
ax[0,0].set_xlabel("Incubation Days")
ax[0,0].set_title("A. Gamma PDFs")
ax[1,1].set_axis_off()
plt.suptitle("TN Approach vs Direct Sampling")
plt.tight_layout()
plt.savefig("posterior_summary_comparison.png", dpi=600)
plt.show()
plt.close()
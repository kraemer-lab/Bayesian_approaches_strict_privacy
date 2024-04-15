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
dfs_sorted = [pd.read_csv(paths[i]) for i in range(len(paths))]

df_all = pd.concat(dfs_sorted)

left_lower = dfs_sorted[0].left_lower.values
left_upper = dfs_sorted[0].left_upper.values
right_lower = dfs_sorted[0].right_lower.values
right_upper = dfs_sorted[0].right_upper.values

left_interval = np.array([[x,y] for x,y in zip(left_lower, right_lower) if x!=y])
left_exact = np.array([[x] for x,y in zip(left_lower, right_lower) if x==y])

right_interval = np.array([[x,y] for x,y in zip(left_upper, right_upper) if x!=y])
right_exact = np.array([[x] for x,y in zip(left_upper, right_upper) if x==y])


def gamma_cdf(x, alpha, beta):
    return pt.gammainc(alpha, x*beta) #/ pt.gamma(alpha) 

def censored(name, alpha, beta, lower, upper):
    L = gamma_cdf(lower, alpha, beta)
    U = gamma_cdf(upper, alpha, beta)
    return pt.log(U - L)

with pm.Model() as mod:
    alpha = pm.TruncatedNormal('alpha', 0, 10, lower=1, upper=30) 
    beta = pm.TruncatedNormal('beta', 0, 10, lower=1, upper=2)
    mu = pm.Deterministic('mu', alpha/beta ) #incubation period mean
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD

    # latent likelihood of 'inexact' incubation periods
    wl = pm.Potential('wl', censored('censoredl2', alpha, beta, 
                                               left_interval[:,1],
                                               left_interval[:,0]))
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    yl = pm.Gamma("yl", alpha, beta, observed=left_exact[:,0])
    
    # latent likelihood of 'inexact' incubation periods
    wr = pm.Potential('wr', censored('censoredr', alpha, beta, 
                                               right_interval[:,1],
                                               right_interval[:,0]))
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    yr = pm.Gamma("yr", alpha, beta, observed=right_exact[:,0])
    ppc = pm.sample_prior_predictive(1000, random_seed=27)
    
mu_pri = az.extract(ppc.prior)['mu'].values
alp_pri = az.extract(ppc.prior)['alpha'].values
bet_pri = az.extract(ppc.prior)['beta'].values
plt.hist(mu_pri.T, bins=100)

with mod:
    idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                      random_seed=27)

del idata.observed_data
del idata.sample_stats

idxs = np.arange(len(dfs_sorted))

idatas = [idata]

# names for saving idata by site and ordered number (idata01, idata02...etc)
names = ["site"+str(idxs[0]+1)+"_idata01.nc"]

# save inference data from model
az.to_netcdf(idata, "./mvn_approx/"+names[0])

## loop over datasets, save and load idata to update prior
def norm_cdf(x, mu, sigma):
    return 0.5 * (1 + pt.erf( (x-mu) / (sigma*pt.sqrt(2)) )) 

def censored(name, mu, sigma, lower, upper):
    L = norm_cdf(lower, mu, sigma)
    U = norm_cdf(upper, mu, sigma)
    return pt.log(U - L)

for i in range(len(dfs_sorted[1:])):
    if i+1 < 10:
        n = "0"+str(i+1)
    else:
        n = str(i+1)
    name = "site"+str(idxs[i]+1)+"_idata"+n+".nc"
    names.append(name)
    df = dfs_sorted[1:][i]
    pre_mu = az.extract(idatas[i].posterior)['mu'].values #mu from site s-1
    pre_sigma = az.extract(idatas[i].posterior)['sigma'].values #sigma from site s-1
    with pm.Model() as mod:
        prior = make_prior(idatas[i], var_names=['mu', 'sigma'])
        mu = prior['mu'] #* pre_mu.std() + pre_mu.mean()
        sigma = prior['sigma'] #* pre_mu.std() + pre_mu.mean() 
        wl = pm.Potential('wl', censored('censoredl2', mu, sigma, 
                                                   left_interval[:,1],
                                                   left_interval[:,0]))
        if len(left_exact > 0):
            yl = pm.Gamma("yl", mu=mu, sigma=sigma, observed=left_exact[:,0])
        wr = pm.Potential('wr', censored('censoredr', mu, sigma, 
                                                   right_interval[:,1],
                                                   right_interval[:,0]))
        if len(right_exact > 0):
            yr = pm.Gamma("yr", mu=mu, sigma=sigma, observed=right_exact[:,0])
        idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler='numpyro', 
                          random_seed=27)
    if len(left_exact > 0) or len(right_exact > 0):
        del idata.observed_data
    del idata.sample_stats
    az.to_netcdf(idata, "./mvn_approx/"+names[1:][i])
    idata = az.from_netcdf("./mvn_approx/"+names[1:][i])
    idatas.append(idata)

##save summary of latest sampled site
summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("mvn_approx_summary.csv")

pos_mu = az.extract(idata)['mu'].values
pos_sig = az.extract(idata)['sigma'].values


### Sample all data directly
def gamma_cdf(x, alpha, beta):
    return pt.gammainc(alpha, x*beta) #/ pt.gamma(alpha) 

def censored(name, alpha, beta, lower, upper):
    L = gamma_cdf(lower, alpha, beta)
    U = gamma_cdf(upper, alpha, beta)
    return pt.log(U - L)

left_lower = df_all.left_lower.values
left_upper = df_all.left_upper.values
right_lower = df_all.right_lower.values
right_upper = df_all.right_upper.values

left_interval = np.array([[x,y] for x,y in zip(left_lower, right_lower) if x!=y])
left_exact = np.array([[x] for x,y in zip(left_lower, right_lower) if x==y])

right_interval = np.array([[x,y] for x,y in zip(left_upper, right_upper) if x!=y])
right_exact = np.array([[x] for x,y in zip(left_upper, right_upper) if x==y])
with pm.Model() as mod:
    alpha = pm.TruncatedNormal('alpha', 0, 10, lower=1, upper=30) 
    beta = pm.TruncatedNormal('beta', 0, 10, lower=1, upper=2)
    mu = pm.Deterministic('mu', alpha/beta ) #incubation period mean
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD

    # latent likelihood of 'inexact' incubation periods
    wl = pm.Potential('wl', censored('censoredl', alpha, beta, 
                                               left_interval[:,1],
                                               left_interval[:,0]))
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    yl = pm.Gamma("yl", alpha, beta, observed=left_exact[:,0])
    
    # latent likelihood of 'inexact' incubation periods
    wr = pm.Potential('wr', censored('censoredr', alpha, beta, 
                                               right_interval[:,1],
                                               right_interval[:,0]))
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    yr = pm.Gamma("yr", alpha, beta, observed=right_exact[:,0])

with mod:
    idata = pm.sample(2000, tune=2000, nuts_sampler='numpyro', random_seed=27)

pos_mu_all = az.extract(idata)['mu'].values
pos_sig_all = az.extract(idata)['sigma'].values


####compute hellinger indices
sqrt_part = np.sqrt((2*pos_mu_all.std() * pos_mu.std()) / (pos_mu_all.std()**2 + 
                                                               pos_mu.std()**2))
exp_part = np.exp(-0.25 * (pos_mu_all.mean() - pos_mu.mean())**2 / (pos_mu_all.std()**2 + 
                                                                        pos_mu.std()**2))
H2 = np.round(1 - sqrt_part * exp_part, 2)


x = np.array([np.arange(30, step=0.1) for i in range(pos_mu.shape[0])]).T #30 days


cdf = sp.stats.gamma.cdf(x, pos_mu**2 / pos_sig**2) 

cdf_all = sp.stats.gamma.cdf(x, pos_mu_all**2 / pos_sig_all**2) 


### compute Gamma pdfs
pdf = sp.stats.gamma.pdf(x, pos_mu**2 / pos_sig**2) 

pdf_all = sp.stats.gamma.pdf(x, pos_mu_all**2 / pos_sig_all**2) 

# sqrt_part = np.sqrt((2*pdf_all.std() * pdf.std()) / (pdf_all.std()**2 + pdf.std()**2))
# exp_part = np.exp(-0.25 * (pdf_all.mean() - pdf.mean())**2 / (pdf_all.std()**2 + pdf.std()**2))
# H2_pdf = np.round(1 - sqrt_part * exp_part, 4)


## Plot comparison between direct sampling and posterior update
x2 = np.arange(30, step=0.1)
fig, ax = plt.subplots(2,2, figsize=(12,8))
sns.kdeplot(pos_mu_all, color='forestgreen', linewidth=3, ax=ax[1,0])
sns.kdeplot(pos_mu, color='purple', linestyle='--', linewidth=3, ax=ax[1,0])
ax[1,0].text(6.2, 2, "H² = "+str(H2))
ax[1,0].set_axisbelow(True)
ax[1,0].grid(alpha=0.2)
ax[1,0].set_ylabel("Density")
ax[1,0].set_xlabel("Incubation Days")
ax[1,0].set_title("C. Posterior Distributions (μ)")
ax[0,1].plot(x2, cdf_all.mean(axis=1), color='forestgreen', linewidth=3, label="Direct Sampling")
ax[0,1].plot(x2, cdf.mean(axis=1), color='purple', linestyle="--", linewidth=3, label="TN Approx.")
ax[0,1].legend()
ax[0,1].set_axisbelow(True)
ax[0,1].grid(alpha=0.2)
ax[0,1].set_ylabel("Probability")
ax[0,1].set_xlabel("Incubation Days")
ax[0,1].set_title("B. Gamma CDFs")
ax[0,0].plot(x2, pdf_all.mean(axis=1), color='forestgreen', linewidth=3, label="Direct Sampling")
ax[0,0].plot(x2, pdf.mean(axis=1), color='purple', linestyle="--", linewidth=3, label="TN Approx.")
#ax[0,0].legend()
ax[0,0].set_axisbelow(True)
ax[0,0].grid(alpha=0.2)
ax[0,0].set_ylabel("Density")
ax[0,0].set_xlabel("A. Incubation Days")
ax[0,0].set_title("Gamma PDFs")
ax[1,1].set_axis_off()
plt.suptitle("MvN Approach vs Direct Sampling")
plt.tight_layout()
plt.savefig("mvn_approach_comparison.png", dpi=600)
plt.show()
plt.close()
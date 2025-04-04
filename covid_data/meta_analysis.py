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
import numpyro

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


# # save inference data from model
# az.to_netcdf(idata, "./posterior_summary/"+names[0])

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

pos_mu = az.extract(idata_meta)['theta'].values
pos_sig = az.extract(idata_meta)['sigma'].values #to assess between study variability

# Extract posterior means and credible intervals for theta
theta_mean = idata_meta.posterior['theta'].mean(dim=('chain', 'draw')).values
theta_ci = az.hdi(idata_meta, hdi_prob=0.9, var_names=['theta']).theta.values

# Forest plot of each site (meta-analysis)
plt.figure(figsize=(8, 6))
plt.errorbar(theta_mean, np.arange(n_sites), xerr=[theta_mean - theta_ci[:, 0], theta_ci[:, 1] - theta_mean],
             fmt='o', capsize=5, label='Site μ')
plt.axvline(x=idata_meta.posterior['mu'].mean(), color='r', linestyle='--', label='Overall μ')
plt.yticks(np.arange(n_sites), [f'Site {i+1}' for i in range(n_sites)])
plt.xlabel('Effect Size')
plt.title(r"$\bf{B.}$ Meta-analysis", loc="left")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("meta_analysis_site_forestplots.png", dpi=300)
plt.show()
plt.close()




# Extract posterior means for alpha and beta

# idata_meta = az.from_netcdf("./meta_analysis_idata.nc")

pos_mu = az.extract(idata_meta)['mu'].values
pos_sig = az.extract(idata_meta)['sigma'].values.mean(axis=0)

# pos_alpha = az.extract(idata_meta)['alpha'].values
# pos_beta = az.extract(idata_meta)['beta'].values

mu_meta = az.extract(idata_meta)['mu'].values
sig_meta = az.extract(idata_meta)['sigma'].values
tau_meta = az.extract(idata_meta)['tau'].values
the_meta = az.extract(idata_meta)['theta'].values
# pos_alpha = the_meta.mean(axis=0)**2 / sig_meta.mean(axis=0)**2
# pos_beta = the_meta.mean(axis=0) / sig_meta.mean(axis=0)**2

pos_alpha = pos_mu**2 / pos_sig**2 
pos_beta = pos_mu / pos_sig**2 

# Calculate mean alpha and beta across posterior samples
mean_alpha = np.median(pos_alpha)
mean_beta = np.median(pos_beta)

x = np.linspace(0, 20, 1000)
pdf = gamma.pdf(x, a=mean_alpha, scale=1/mean_beta)  # Theoretical gamma PDF

gam_dist = gamma(a=mean_alpha, scale=1/mean_beta)
gam_samples = gam_dist.rvs(size=8000)  # Draw 10,000 random samples
pdf5, pdf95 = az.hdi(gam_samples, hdi_prob=0.9)  # 90% HDI

mode_pos = (pos_alpha - 1) / pos_beta

median_pdf_pos = gamma.median(pos_alpha, scale=1/pos_beta)

median_pdf = np.median(median_pdf_pos)
med5, med95 = list(az.hdi(median_pdf_pos, hdi_prob=0.9))

mode_pdf = np.median(mode_pos)
mod5, mod95 = list(az.hdi(mode_pos, hdi_prob=0.9))

pos_mean = pos_mu.mean()
mea5, mea95 = list(az.hdi(pos_mu, hdi_prob=0.9))

# Plot the pooled gamma distribution using posterior information
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'r-', lw=2, label=f'Gamma Distribution (α={mean_alpha:.2f}, β={mean_beta:.2f})')
plt.fill_between(x, pdf, where=(x >= pdf5) & (x <= pdf95), color='gray', alpha=0.3, label=f'90% HDI: {pdf5:.2f} - {pdf95:.2f} days')
plt.axvline(pos_mean, c='k', ls='--', label=f'Mean = {pos_mean:.2f} [{mea5:.2f}, {mea95:.2f}] days')
plt.axvline(median_pdf, c='k', ls=':', label=f'Median = {median_pdf:.2f} [{med5:.2f}, {med95:.2f}] days')
plt.axvline(mode_pdf, c='k', ls='-.', label=f'Mode = {mode_pdf:.2f} [{mod5:.2f}, {mod95:.2f}] days')
plt.title(r"$\bf{B.}$ Meta-analysis Gamma Distribution (COVID-19)", loc="left", size=20)
plt.xlabel("Incubation Period (Days)", size=18)
plt.ylabel("Density")
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('meta_analysis_summary.png', dpi=600)
plt.show()
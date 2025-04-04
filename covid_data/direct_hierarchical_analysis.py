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

# read data
paths = glob.glob("./chunked_data/*")
paths = np.sort(paths)

dfs_sorted = [pd.read_csv(paths[i]) for i in range(len(paths))]


df_all = pd.concat(dfs_sorted)

n_obs = np.array([len(df) for df in dfs_sorted])


# Factorize SiteIdx to create integer indices
idxs = pd.factorize(df_all.SiteIdx)[0]
sites_id = pd.unique(idxs)
n_sites = len(sites_id)

# Extract values from the DataFrame
left_lower = df_all.left_lower.values
left_upper = df_all.left_upper.values
right_lower = df_all.right_lower.values
right_upper = df_all.right_upper.values

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

# Extract the corresponding indices for exact and inexact data
left_interval_idxs = idxs[left_interval_indices]
left_exact_idxs = idxs[left_exact_indices]

right_interval_idxs = idxs[right_interval_indices]
right_exact_idxs = idxs[right_exact_indices]

# Define the Gamma CDF and censored likelihood

def censored(name, alpha, beta, lower, upper):
    L = pt.gammainc(alpha, lower*beta)
    U = pt.gammainc(alpha, upper*beta)
    return pt.log(U - L)

weights = pt.sqrt(1/n_obs)

# Build the PyMC model
with pm.Model() as mod:
        
    alpha_s = pm.HalfNormal("alpha_s", 0.5)
    beta_s = pm.HalfNormal("beta_s", 0.5)
    
    alpha_z = pm.Normal('alpha_z', 0, 0.1, shape=n_sites) 
    beta_z = pm.Normal('beta_z', 0, 0.1, shape=n_sites)
    
    # Transformed parameters (ensuring positivity)
    alpha = pm.Deterministic("alpha",  pt.exp(pt.log(4) + alpha_s * alpha_z))  # exp transform
    beta = pm.Deterministic("beta",  pt.exp(pt.log(0.66) + beta_s * beta_z))     # exp transform
   
    mu = pm.Deterministic('mu', alpha/beta) #incubation period mean
    
    sigma = pm.Deterministic('sigma', pt.sqrt(alpha/beta**2)) #incubation period SD
            
    # Latent likelihood for inexact intervals (left)
    wl = pm.Potential('wl', censored('censoredl', alpha[left_interval_idxs], beta[left_interval_idxs], 
                                     left_interval[:, 1], left_interval[:, 0]))
    
    # Likelihood for exact intervals (left)
    yl = pm.Gamma("yl", alpha=alpha[left_exact_idxs], beta=beta[left_exact_idxs], observed=left_exact)
    
    # Latent likelihood for inexact intervals (right)
    wr = pm.Potential('wr', censored('censoredr', alpha[right_interval_idxs], beta[right_interval_idxs], 
                                     right_interval[:, 1], right_interval[:, 0]))
    
    # Likelihood for exact intervals (right)
    yr = pm.Gamma("yr", alpha=alpha[right_exact_idxs], beta=beta[right_exact_idxs], observed=right_exact)

# Sample from the model
with mod:
    idata_dir = pm.sample(2000, tune=2000, nuts_sampler='numpyro', random_seed=27)
    
try:
    del idata_dir.observed_data
except:
    pass
try:
    del idata_dir.sample_stats
except:
    pass

summ = az.summary(idata_dir, hdi_prob=0.9)
summ.to_csv("direct_analysis_summary.csv")

az.to_netcdf(idata_dir, "./direct_analysis_idata.nc")

# idata_dir = az.from_netcdf("./direct_analysis_idata.nc")

pos_mu_all = az.extract(idata_dir)['mu'].values.mean(axis=0)
pos_sig_all = az.extract(idata_dir)['sigma'].values.mean(axis=0)

# Forest plot each site (full data hierarchical model) 
theta_mean = idata_dir.posterior['mu'].mean(dim=('chain', 'draw')).values
theta_ci = az.hdi(idata_dir, hdi_prob=0.9, var_names=['mu']).mu.values
plt.figure(figsize=(8, 6))
plt.errorbar(theta_mean, np.arange(n_sites), xerr=[theta_mean - theta_ci[:, 0], theta_ci[:, 1] - theta_mean],
             fmt='o', capsize=5, label='Site μ')
plt.axvline(x=idata_dir.posterior['mu'].mean(), color='r', linestyle='--', label='Overal μ')
plt.yticks(np.arange(n_sites), [f'Site {i+1}' for i in range(n_sites)])
plt.xlabel('Effect Size')
plt.title(r"$\bf{C.}$ Direct Sampling", loc="left")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("direct_all_site_forestplots.png", dpi=300)
plt.show()
plt.close()


# idata_dir = az.from_netcdf("./direct_analysis_idata.nc")

# Extract posterior means for alpha and beta

pos_alpha = az.extract(idata_dir)['alpha'].values.mean(axis=0)
pos_beta = az.extract(idata_dir)['beta'].values.mean(axis=0)

# Calculate mean alpha and beta across posterior samples
mean_alpha = np.median(pos_alpha)
mean_beta = np.median(pos_beta)

x = np.linspace(0, 20, 1000)
pdf = gamma.pdf(x, a=mean_alpha, scale=1/mean_beta)  # Theoretical gamma PDF

gam_dist = gamma(a=mean_alpha, scale=1/mean_beta)
gam_samples = gam_dist.rvs(size=8000)  # Draw 10,000 random samples
pdf5, pdf95 = az.hdi(gam_samples, hdi_prob=0.9)  # 90% HDI

mode_pos = (pos_alpha - 1) / pos_beta

median_pdf_pos = gamma.median(pos_alpha.mean(), scale=1/pos_beta)

median_pdf = np.median(median_pdf_pos)
med5, med95 = list(az.hdi(median_pdf_pos, hdi_prob=0.9))

mode_pdf = np.median(mode_pos)
mod5, mod95 = list(az.hdi(mode_pos, hdi_prob=0.9))

pos_mean = pos_mu_all.mean()
mea5, mea95 = list(az.hdi(pos_mu_all, hdi_prob=0.9))


# Plot the pooled gamma distribution using posterior information
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'r-', lw=2, label=f'Gamma Distribution (α={mean_alpha:.2f}, β={mean_beta:.2f})')
plt.fill_between(x, pdf, where=(x >= pdf5) & (x <= pdf95), color='gray', alpha=0.3, label=f'90% HDI: {pdf5:.2f} - {pdf95:.2f} days')
plt.axvline(pos_mean, c='k', ls='--', label=f'Mean = {pos_mean:.2f} [{mea5:.2f}, {mea95:.2f}] days')
plt.axvline(median_pdf, c='k', ls=':', label=f'Median = {median_pdf:.2f} [{med5:.2f}, {med95:.2f}] days')
plt.axvline(mode_pdf, c='k', ls='-.', label=f'Mode = {mode_pdf:.2f} [{mod5:.2f}, {mod95:.2f}] days')
plt.title(r"$\bf{C.}$ Direct Sampling Gamma Distribution (COVID-19)", loc="left", size=20)
plt.xlabel("Incubation Period (Days)", size=18)
plt.ylabel("Density")
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('direct_hirarchical_summary.png', dpi=600)
plt.show()
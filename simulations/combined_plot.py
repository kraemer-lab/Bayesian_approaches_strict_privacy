# -*- coding: utf-8 -*-
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import pymc_experimental as pme
make_prior = pme.utils.prior.prior_from_idata
from scipy.stats import gamma
import matplotlib.gridspec as gridspec
import pandas as pd
import glob

#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

np.random.seed(27) # set numpy random seed


idata_mvn = az.from_netcdf("./mvn_approx_idata.nc")
idata_meta = az.from_netcdf("./meta_analysis_idata.nc")
idata_dir = az.from_netcdf("./direct_analysis_idata.nc")

dir_summ = az.summary(idata_dir, hdi_prob=0.9)
dir_summ.to_csv("direct_analysis_summary.csv")

mvn_summ = az.summary(idata_mvn, hdi_prob=0.9)
mvn_summ.to_csv("mvn_approx_summary.csv")

meta_summ = az.summary(idata_meta, hdi_prob=0.9)
meta_summ.to_csv("meta_analysis_summary.csv")


mvn_a = az.extract(idata_mvn)['alpha'].values
mvn_b = az.extract(idata_mvn)['beta'].values

meta_a = az.extract(idata_meta)['alpha'].values
meta_b = az.extract(idata_meta)['beta'].values

dir_a = az.extract(idata_dir)['alpha'].values.mean(axis=0)
dir_b = az.extract(idata_dir)['beta'].values.mean(axis=0)

mu_mvn = az.extract(idata_mvn)['mu'].values
sig_mvn = az.extract(idata_mvn)['sigma'].values

mu_mvn = az.extract(idata_mvn)['mu'].values
sig_mvn = az.extract(idata_mvn)['sigma'].values

mu_meta = az.extract(idata_meta)['mu'].values
sig_meta = az.extract(idata_meta)['sigma'].values
tau_meta = az.extract(idata_meta)['tau'].values
the_meta = az.extract(idata_meta)['theta'].values

mu_dir = az.extract(idata_dir)['mu'].values
sig_dir = az.extract(idata_dir)['sigma'].values

dir_a = mu_dir.mean(axis=0)**2 / sig_dir.mean(axis=0)**2
dir_b = mu_dir.mean(axis=0) / sig_dir.mean(axis=0)**2

meta_a = the_meta.mean(axis=0)**2 / sig_meta.mean(axis=0)**2
meta_b = the_meta.mean(axis=0) / sig_meta.mean(axis=0)**2

mvn_a = mu_mvn**2 / sig_mvn**2
mvn_b = mu_mvn / sig_mvn**2

#### Compute overlap and divergence indices

def overlap_coefficient_gamma(a1, b1, a2, b2):

    # Check for valid parameters
    if a1 <= 0 or b1 <= 0 or a2 <= 0 or b2 <= 0:
        raise ValueError("Invalid parameters: k and theta must be positive.")

    # Define the range of x values for integration
    x = np.linspace(0, 30, 10000)

    # Compute the PDFs of the two gamma distributions
    pdf1 = gamma.pdf(x, a=a1, scale=1/b1)
    pdf2 = gamma.pdf(x, a=a2, scale=1/b2)

    # Compute the minimum of the two PDFs at each point
    min_pdf = np.minimum(pdf1, pdf2)

    # Integrate the minimum PDF to get the overlap coefficient
    overlap = np.trapezoid(min_pdf, x)

    return overlap
    


ovl_mvn_pdf = np.round(overlap_coefficient_gamma(mvn_a.mean(), mvn_b.mean(), 
                                              dir_a.mean(), dir_b.mean()), 2)

ovl_meta_pdf = np.round(overlap_coefficient_gamma(meta_a.mean(), meta_b.mean(), 
                                              dir_a.mean(), dir_b.mean()), 2)


#### Compute CDFs
x = np.arange(30, step=0.1) #30 days

x2 = np.array([np.arange(30, step=0.1) for i in range(mu_dir.T.shape[0])]).T #30 days


cdf_dir = sp.stats.gamma.cdf(x2, dir_a, scale=1/dir_b) 
hdi_dir_cdf = az.hdi(cdf_dir.T, hdi_prob=0.9).T

cdf_meta = sp.stats.gamma.cdf(x2, meta_a, scale=1/meta_b) 
hdi_meta_cdf = az.hdi(cdf_meta.T, hdi_prob=0.9).T

cdf_mvn = sp.stats.gamma.cdf(x2, mvn_a, scale=1/mvn_b) 
hdi_mvn_cdf = az.hdi(cdf_mvn.T, hdi_prob=0.9).T

pdf_dir = sp.stats.gamma.pdf(x2, dir_a, scale=1/dir_b) 
hdi_dir_pdf = az.hdi(pdf_dir.T, hdi_prob=0.9).T

pdf_meta = sp.stats.gamma.pdf(x2, meta_a, scale=1/meta_b) 
hdi_meta_pdf = az.hdi(pdf_meta.T, hdi_prob=0.9).T

pdf_mvn = sp.stats.gamma.pdf(x2, mvn_a, scale=1/mvn_b) 
hdi_mvn_pdf = az.hdi(pdf_mvn.T, hdi_prob=0.9).T




# Create the figure and outer grid
fig = plt.figure(figsize=(18, 8))
outer_grid = gridspec.GridSpec(2, 2, wspace=0.15, hspace=0.4)

# Plot in the first 3 subplots (ax[0,0], ax[0,1], ax[1,0])
ax1 = fig.add_subplot(outer_grid[0, 0])
ax1.plot(x, pdf_dir.mean(axis=1), c='steelblue', lw=4, label="Direct Sampling")
ax1.plot(x, pdf_mvn.mean(axis=1), c='purple', ls="-.", lw=4, label="MvN-approx")
ax1.plot(x, pdf_meta.mean(axis=1), c='orangered', ls="--", lw=4, label="Meta-analysis")
ax1.text(mu_dir.mean() + 5, 0.1, r"$OVL_{MvN}$ = " + str(ovl_mvn_pdf), size=14)
ax1.text(mu_dir.mean() + 5, 0.08, r"$OVL_{Meta}$ = " + str(ovl_meta_pdf), size=14)
ax1.set_axisbelow(True)
ax1.grid(alpha=0.3)
ax1.set_ylabel("Density")
ax1.set_xlabel("Days", size=15)
ax1.set_title(r"$\bf{A.}$ Gamma PDFs (mean)", loc="left")
ax1.set_xticks(np.arange(min(x), max(x) + 1, 1))
ax1.set_xlim(0,22)

ax2 = fig.add_subplot(outer_grid[0, 1])
ax2.plot(x, cdf_dir.mean(axis=1), c='steelblue', lw=4, label="Direct Sampling")
ax2.plot(x, cdf_mvn.mean(axis=1), c='purple', ls="-.", lw=4, label="MvN-approx")
ax2.plot(x, cdf_meta.mean(axis=1), c='orangered', ls="--", lw=4, label="Meta-analysis")
ax2.legend(loc="lower right", fontsize=14)
ax2.set_axisbelow(True)
ax2.grid(alpha=0.3)
ax2.set_ylabel("Probability")
ax2.set_xlabel("Days", size=15)
ax2.set_title(r"$\bf{B.}$ Gamma CDFs (mean)", loc="left")
ax2.set_xticks(np.arange(min(x), max(x) + 1, 1))
ax2.set_xlim(0,22)

ax3 = fig.add_subplot(outer_grid[1, 0])
sns.kdeplot(mu_dir.mean(axis=0), c='steelblue', lw=4, ax=ax3, bw_adjust=1)
sns.kdeplot(mu_mvn, c='purple', ls='-.', lw=4, ax=ax3, bw_adjust=1)
sns.kdeplot(mu_meta, c='orangered', ls='--', lw=4, ax=ax3, bw_adjust=1)
ax3.set_axisbelow(True)
ax3.grid(alpha=0.3)
ax3.set_ylabel("Density")
ax3.set_xlabel("Days", size=15)
ax3.set_title(r"$\bf{C.}$ Posterior Distributions (μ)", loc="left")
ax3.set_xticks(np.arange(2, 12 + 1, 1))
ax3.set_xlim(4,12)

# Divide the space of ax[1, 1] into 3 separate plots
inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[1, 1], wspace=0.05,)

ax4 = fig.add_subplot(inner_grid[0])
ax4.plot(x, cdf_dir.mean(axis=1), color='steelblue', linewidth=1, label="Mean")
ax4.fill_between(x, hdi_dir_cdf[0], hdi_dir_cdf[1], color='steelblue', alpha=0.2, label="90% HDI")
ax4.set_ylabel("Probability")
ax4.legend(fontsize=11, loc="lower right")
ax4.grid(alpha=0.3)
ax4.set_title(r"$\bf{D.}$ Direct Sampling", size=14, loc="left")
ax4.set_xlim(0,22)
ax4.set_xticks(np.arange(0, 25, 5))

ax5 = fig.add_subplot(inner_grid[1])
ax5.plot(x, cdf_mvn.mean(axis=1), color='purple', linewidth=1, ls="-.", label="Mean")
ax5.fill_between(x, hdi_mvn_cdf[0], hdi_mvn_cdf[1], color='purple', alpha=0.2, label="90% HDI")
ax5.set_yticks([]) 
ax5.legend(fontsize=11, loc="lower right")
ax5.grid(alpha=0.3)
ax5.set_title(r"$\bf{E.}$ MvN-approx.", size=14, loc="left")
ax5.set_xlabel("Days", size=15)
ax5.set_xlim(0,22)
ax5.set_xticks(np.arange(0, 25, 5))

ax6 = fig.add_subplot(inner_grid[2])
ax6.plot(x, cdf_meta.mean(axis=1), color='orangered', linewidth=1, ls="--", label="Mean")
ax6.fill_between(x, hdi_meta_cdf[0], hdi_meta_cdf[1], color='orangered', alpha=0.2, label="90% HDI")
ax6.set_yticks([])  
ax6.legend(fontsize=11, loc="lower right")
ax6.grid(alpha=0.3)
ax6.set_title(r"$\bf{F.}$ Meta-analysis", size=14, loc="left")
ax6.set_xlim(0,22)
ax6.set_xticks(np.arange(0, 25, 5))

# Add a suptitle and save the figure
plt.suptitle("Mpox Simulations Incubation Period Estimation (Comparison)", size=20, y=0.99)
plt.tight_layout()
plt.savefig("direct_mvn_meta_comparison.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()


################## Plot Extra Summaries

alphas = [dir_a, mvn_a, meta_a]
betas = [dir_b, mvn_b, meta_b]

fig, ax = plt.subplots(2,2, figsize=(20,12))
ax = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]

names = [r"$\bf{A.}$ Direct Sampling Gamma Distribution (Mpox Simulations)",
         r"$\bf{B.}$ MvN-approx Gamma Distribution (Mpox Simulations)",
         r"$\bf{C.}$ Meta-analysis Gamma Distribution (Mpox Simulations)",] 

for i in range(len(ax)-1):
    
    name = names[i]
    
    a = alphas[i]
    b = betas [i]
    
    pdf = sp.stats.gamma.pdf(x2, a, scale=1/b) 
    
    pdf_m = pdf.mean(axis=1)
    pdf5, pdf95 = az.hdi(pdf.T, hdi_prob=0.9).T  # 90% HDI
    
    gam_dist = gamma(a=a, scale=1/b)
    gam_samples = gam_dist.rvs(size=8000)  # Draw 8,000 random samples
    h5, h95 = az.hdi(gam_samples, hdi_prob=0.9)  # 90% HDI
    
    mode_pos = (a - 1) / b
    
    median_pdf_pos = gamma.median(a, scale=1/b)
    
    median_pdf = np.median(median_pdf_pos)
    med5, med95 = list(az.hdi(median_pdf_pos, hdi_prob=0.9))
    
    mode_pdf = np.median(mode_pos)
    mod5, mod95 = list(az.hdi(mode_pos, hdi_prob=0.9))
    
    pos_mean = (a/b).mean()
    mea5, mea95 = list(az.hdi(a/b, hdi_prob=0.9))
    
    a_mean = a.mean()
    b_mean = b.mean()
    
    # Plot the pooled gamma distribution using posterior information
    ax[i].plot(x, pdf_m, 'r-', lw=2.5, label=f'PDF (α={a_mean:.2f}, β={b_mean:.2f})')
    ax[i].fill_between(x, pdf5, pdf95, color='crimson', alpha=0.3, label="PDF 90% HDI")
    ax[i].fill_between(x, np.median(pdf, axis=1), where=(x >= h5) & (x <= h95), color='gray', alpha=0.3, label=f'90% HDI: {h5:.2f} - {h95:.2f} days')
    ax[i].axvline(pos_mean, c='k', ls='--', label=f'Mean = {pos_mean:.2f} [{mea5:.2f}, {mea95:.2f}] days')
    ax[i].axvline(median_pdf, c='k', ls=':', label=f'Median = {median_pdf:.2f} [{med5:.2f}, {med95:.2f}] days')
    ax[i].axvline(mode_pdf, c='k', ls='-.', label=f'Mode = {mode_pdf:.2f} [{mod5:.2f}, {mod95:.2f}] days')
    ax[i].set_title(name, loc="left", size=20)
    ax[i].set_xlabel("Incubation Period (Days)", size=18)
    ax[i].set_ylabel("Density")
    ax[i].legend(fontsize=16)
    ax[i].grid(True)
    ax[i].set_xlim(0,25)

ax[-1].set_axis_off()
plt.tight_layout()
plt.savefig('gamma_all_summary.png', dpi=600)
plt.show()



################ Forest plots ################
mvn_idatas = [az.from_netcdf(d) for d in glob.glob("./mvn_approx/*")]
mvn_theta = [az.extract(i)['mu'].values for i in mvn_idatas]
mvn_theta.append(az.extract(idata_mvn)['mu'].values)
mvn_theta = np.array(mvn_theta)

meta_theta = az.extract(idata_meta)['theta'].values

dir_theta = az.extract(idata_dir)['mu'].values

n_sites = len(mvn_theta)

thetas = [dir_theta, mvn_theta, meta_theta]

fig, ax = plt.subplots(2,2, figsize=(12,8))
ax = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]

names = [r"$\bf{A.}$ Direct Sampling (Mpox Simulations)",
         r"$\bf{B.}$ MvN-approx (Mpox Simulations)",
         r"$\bf{C.}$ Meta-analysis (Mpox Simulations)",] 

for i in range(len(ax)-1):
    
    name = names[i]
    
    theta = thetas[i]
    
    # Extract posterior means and credible intervals for theta
    tmean = theta.mean(axis=1)
    thdi = az.hdi(theta.T, hdi_prob=0.9).T
    
    # Forest plot of each site (meta-analysis)
    ax[i].errorbar(tmean, np.arange(n_sites), xerr=[tmean - thdi[0,:], thdi[1,:] - tmean],
                 fmt='o', capsize=5, label='Site μ')
    ax[i].axvline(x=tmean.mean(), color='r', linestyle='--', label='Overall μ')
    ax[i].set_yticks(np.arange(n_sites), [f'Site {i+1}' for i in range(n_sites)])
    ax[i].set_xlabel('Effect Size')
    ax[i].set_title(name, loc="left")
    ax[i].invert_yaxis()
ax[0].legend()
ax[-1].set_axis_off() 
plt.tight_layout()
plt.savefig("summary_forestplots.png", dpi=600)
plt.show()
plt.close()





##### Table summaries

    

# Calculate mean alpha and beta across posterior samples

mean_pos_dir = mu_dir.mean(axis=1)
mean_dir = mu_dir.mean()
mea_dir5, mea_dir95 = list(az.hdi(mean_pos_dir, hdi_prob=0.9))

median_pos_dir = gamma.median(dir_a, scale=1/dir_b)
median_dir = median_pos_dir.mean()
med_dir5, med_dir95 = list(az.hdi(median_pos_dir, hdi_prob=0.9))

mode_pos_dir = (dir_a - 1) / dir_b
mode_dir = mode_pos_dir.mean()
mod_dir5, mod_dir95 = list(az.hdi(mode_pos_dir, hdi_prob=0.9))

mean_pos_mvn = mu_mvn
mean_mvn = mu_mvn.mean()
mea_mvn5, mea_mvn95 = list(az.hdi(mean_pos_mvn, hdi_prob=0.9))

median_pos_mvn = gamma.median(mvn_a, scale=1/mvn_b)
median_mvn = median_pos_mvn.mean()
med_mvn5, med_mvn95 = list(az.hdi(median_pos_mvn, hdi_prob=0.9))

mode_pos_mvn = (mvn_a - 1) / mvn_b
mode_mvn = mode_pos_mvn.mean()
mod_mvn5, mod_mvn95 = list(az.hdi(mode_pos_mvn, hdi_prob=0.9))

mean_pos_meta = mu_meta
mean_meta = mu_meta.mean()
mea_meta5, mea_meta95 = list(az.hdi(mean_pos_meta, hdi_prob=0.9))

median_pos_meta = gamma.median(meta_a, scale=1/meta_b)
median_meta = median_pos_meta.mean()
med_meta5, med_meta95 = list(az.hdi(median_pos_meta, hdi_prob=0.9))

mode_pos_meta = (meta_a - 1) / meta_b
mode_meta = mode_pos_meta.mean()
mod_meta5, mod_meta95 = list(az.hdi(mode_pos_meta, hdi_prob=0.9))



# Define methods and statistics
methods = ['Direct Sampling', 'MvN-approx.', 'Meta-analysis']
stats = ['mean', 'sd', 'hdi']  # Now only 3 columns per method

# Create MultiIndex columns
columns = pd.MultiIndex.from_product([methods, stats])

# Initialize DataFrame with statistics as rows
gamma_stats = ['Mean', 'Median', 'Mode']
df_gamma = pd.DataFrame(index=gamma_stats, columns=columns)

# Fill data for each method
for method in methods:
    # Select variables based on method
    if method == 'Direct Sampling':
        mean_val, mean_sd = mean_dir, np.std(mean_pos_dir)
        median_val, median_sd = median_dir, np.std(median_pos_dir)
        mode_val, mode_sd = mode_dir, np.std(mode_pos_dir)
        hdi_mean = [mea_dir5, mea_dir95]
        hdi_median = [med_dir5, med_dir95]
        hdi_mode = [mod_dir5, mod_dir95]
    elif method == 'MvN-approx.':
        mean_val, mean_sd = mean_mvn, np.std(mean_pos_mvn)
        median_val, median_sd = median_mvn, np.std(median_pos_mvn)
        mode_val, mode_sd = mode_mvn, np.std(mode_pos_mvn)
        hdi_mean = [mea_mvn5, mea_mvn95]
        hdi_median = [med_mvn5, med_mvn95]
        hdi_mode = [mod_mvn5, mod_mvn95]
    else:  # Meta-analysis
        mean_val, mean_sd = mean_meta, np.std(mean_pos_meta)
        median_val, median_sd = median_meta, np.std(median_pos_meta)
        mode_val, mode_sd = mode_meta, np.std(mode_pos_meta)
        hdi_mean = [mea_meta5, mea_meta95]
        hdi_median = [med_meta5, med_meta95]
        hdi_mode = [mod_meta5, mod_meta95]
    
    # Assign values to DataFrame with combined HDI
    df_gamma.loc['Mean', (method, 'mean')] = f"{mean_val:.2f}"
    df_gamma.loc['Mean', (method, 'sd')] = f"{mean_sd:.2f}"
    df_gamma.loc['Mean', (method, 'hdi')] = f"[{hdi_mean[0]:.2f}, {hdi_mean[1]:.2f}]"
    
    df_gamma.loc['Median', (method, 'mean')] = f"{median_val:.2f}"
    df_gamma.loc['Median', (method, 'sd')] = f"{median_sd:.2f}"
    df_gamma.loc['Median', (method, 'hdi')] = f"[{hdi_median[0]:.2f}, {hdi_median[1]:.2f}]"
    
    df_gamma.loc['Mode', (method, 'mean')] = f"{mode_val:.2f}"
    df_gamma.loc['Mode', (method, 'sd')] = f"{mode_sd:.2f}"
    df_gamma.loc['Mode', (method, 'hdi')] = f"[{hdi_mode[0]:.2f}, {hdi_mode[1]:.2f}]"

# Display
print("Gamma Distribution Statistics Comparison")
print("="*80)
print(df_gamma)

# Save to CSV
df_gamma.to_csv("gamma_statistics_comparison.csv")




###Parameter comparisons
gam_dir = gamma(a=dir_a.mean(), scale=1/dir_b.mean())
gam_dir = gam_dir.rvs(size=8000)  # Draw 8,000 random samples

gam_meta = gamma(a=meta_a.mean(), scale=1/meta_b.mean())
gam_meta = gam_meta.rvs(size=8000)  # Draw 8,000 random samples

gam_mvn = gamma(a=mvn_a.mean(), scale=1/mvn_b.mean())
gam_mvn = gam_mvn.rvs(size=8000)  # Draw 8,000 random samples


# Data structure
params = ['alpha', 'beta', 'mu', 'sigma', 'pdf']
methods = ['Direct Sampling', 'MvN-approx.', 'Meta-analysis']
stats = ['mean', 'sd', 'hdi']

# Create MultiIndex columns
columns = pd.MultiIndex.from_product([methods, stats])

# Initialize empty DataFrame
df = pd.DataFrame(index=params, columns=columns)

# Fill data
for param in params:
    for method in methods:
        # Select variables based on method
        if method == 'Direct Sampling':
            var_dict = {'alpha': dir_a, 'beta': dir_b, 'mu': mu_dir.mean(axis=1), 
                       'sigma': sig_dir.mean(axis=1), 'pdf': gam_dir}
        elif method == 'MvN-approx.':
            var_dict = {'alpha': mvn_a, 'beta': mvn_b, 'mu': mu_mvn,
                       'sigma': sig_mvn, 'pdf': gam_mvn}
        else:
            var_dict = {'alpha': meta_a, 'beta': meta_b, 'mu': mu_meta,
                       'sigma': sig_meta.mean(axis=1), 'pdf': gam_meta}
        
        # Calculate statistics
        data = var_dict[param]
        df.loc[param, (method, 'mean')] = np.mean(data)
        df.loc[param, (method, 'sd')] = np.std(data)
        hdi = az.hdi(data, hdi_prob=0.9)
        df.loc[param, (method, 'hdi')] = f"[{hdi[0]:.2f}, {hdi[1]:.2f}]"

# Formatting
df = df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

# Display
print("Parameter Statistics Comparison")
print("="*80)
print(df.to_string(float_format="%.2f"))

# Output to files
df.to_csv("parameter_comparison.csv")




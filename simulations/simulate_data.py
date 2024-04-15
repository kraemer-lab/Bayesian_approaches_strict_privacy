# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import glob

#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

np.random.seed(27) # set numpy random seed

#simulate 500 observation and chunk them into 12 groups representing data collection sites (e.g. labs)
#the Poisson distribution represents the count of days from exposure date to illness onset day (i.e. onset date - exposure date)
# mu = pm.TruncatedNormal.dist(9, 3, lower=6, upper=12)
# y = pm.Poisson.dist(mu=mu)

y = pm.NegativeBinomial.dist(mu=9, alpha=10)

#nu = pm.draw(pm.Normal.dist(0, 2), 500, random_seed=27)

data = pm.draw(y, 500, random_seed=27) 
data[data == 0] = 1
data[data > 29] = 28

data2 = data.copy() 

data2 = data2.round(0)


data = data2

chunks = np.array([21, 53, 64, 24, 58, 52, 45, 27, 47, 34, 33, 42])

datas = []
for c in range(len(chunks)):
    d = data2[:chunks[c]]
    datas.append(d)
    data2 = data2[chunks[c]:]
    

# Plot simulation results
plt.hist(data, color='crimson', bins=20, edgecolor='k')
plt.rc('axes', axisbelow=True)
plt.grid(alpha=0.3)
plt.ylabel('Frequency')
plt.xlabel('Incubation days')
plt.title('Simulated incubation period data')
plt.tight_layout()
plt.savefig('simulated_data_all.png', dpi=600)
plt.show()
plt.close()

fig, ax = plt.subplots(3,4, figsize=(20,10))
for d in range(len(datas)):
    da = datas[d]
    if d < 4:
        i=0
        j=d
    if d > 3  and d < 8:
        i=1
        j=d-4
    if d > 7:
        i=2
        j=d-8
    ax[i,j].hist(da, color='crimson', bins=25, edgecolor='k')
    ax[i,j].set_axisbelow(True)
    plt.grid(alpha=0.3)
    ax[i,j].set_ylabel('Frequency')
    ax[i,j].set_xlabel('Incubation days')
    ax[i,j].set_title("Simulated site "+str(d+1))
plt.tight_layout()
plt.savefig('simulated_data_sites.png', dpi=600)
plt.show()
plt.close()


#Save simulated data
dfs = []
i = 0
for d in datas:
    i = i+1
    if i < 9:
        num = "0"+str(i)
    else:
        num = str(i)
    da = pd.DataFrame({'days':d, 'site':np.repeat(i-1, len(d))})
    da.to_csv("./sim_data/data_site"+num+".csv", index=False)
    dfs.append(da)
df = pd.concat(dfs)

#check data works and save ordered data
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

means = [d.days.mean() for d in dfs_sorted]
sds = [d.days.std() for d in dfs_sorted]
dfs_sizes = pd.DataFrame({'site':np.array(idxs)+1, 'sample':np.flip(np.sort(len_dfs)), 
                          'mean':means, 'sd':sds})
dfs_sizes.to_csv("dfs_sizes.csv", index=False)


site_idx = df.site.values
sites = len(df.site.unique())

# Fit model to all data.
with pm.Model() as mod:
    lamz = pm.Normal('lamz', mu=0, sigma=2, shape=sites)
    lam = pm.Normal('lam', mu=0, sigma=2)
    sigma = pm.TruncatedNormal('sigma', mu=0, sigma=2, lower=0)
    mu = pm.Deterministic("mu", pm.math.exp(lam + lamz*sigma))
    alpha = pm.TruncatedNormal('alpha', mu=0, sigma=2, lower=0)
    y = pm.NegativeBinomial('y', mu=mu[site_idx], alpha=alpha, observed=data)
    
with mod:
    idata = pm.sample(2000, tune=2000, chains=4, nuts_sampler='numpyro', target_accept=0.95, random_seed=27)

fix, ax = plt.subplots(5,2, figsize=(10,10))
az.plot_trace(idata, kind='rank_vlines', axes=ax)
ax[0,0].set_title("$λ_z$")
ax[0,1].set_title("$λ_z$")
ax[1,0].set_title("$λ$")
ax[1,1].set_title("$λ$")
ax[2,0].set_title("$σ$")
ax[2,1].set_title("$σ$")
ax[3,0].set_title("$α$")
ax[3,1].set_title("$α$")
ax[4,0].set_title("$μ$")
ax[4,1].set_title("$μ$")
plt.tight_layout()
plt.savefig('simulated_data_traceplot.png', dpi=600)
plt.show()
plt.close()

#save sumamry
summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("summary_simulated_data_model.csv")

#save energy plot and BFMIs
az.plot_energy(idata)
plt.savefig("main_model_energy.png", dpi=300)
plt.close()


##Plot CDF
pos_mu = az.extract(idata)['mu'].values.mean(axis=0)
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
sns.ecdfplot(data, ax=ax, color='k',  label="Observed")
ax.plot(x2, cdf_mean, color='orangered', linestyle="--", label="μ mean")
ax.fill_between(x2, s_low, s_up, color='orangered', alpha=0.2, label="±SD mean")
plt.legend()
ax.set_axisbelow(True)
ax.grid(alpha=0.2)
ax.set_ylabel("Probability")
ax.set_xlabel("Incubation Days")
ax.set_title("Posterior CDF")
plt.tight_layout()
plt.savefig("sim_data_model_posterior.png", dpi=600)
plt.show()
plt.close()

# pos_mu = az.extract(idata)['mu'].values.mean(axis=0)
# h5, h95 = az.hdi(pos_mu, hdi_prob=0.9)
# pos_alp =  az.extract(idata)['alpha'].values
# pos_sd = np.sqrt((pos_mu**2)/pos_alp + pos_mu)
# x = np.array([np.arange(30, step=0.1) for i in range(pos_mu.shape[0])]).T #30 days
# mean_cdfs = sp.stats.nbinom.cdf(x,)
# mean_cdfs = sp.stats.poisson.cdf(x, pos_mu)
# h5, h95 = az.hdi(mean_cdfs.T, hdi_prob=0.9).T
# mme = mean_cdfs.mean(axis=1)
# alp_cdfs = sp.stats.poisson.cdf(x, pos_alp)
# s_low = sp.stats.poisson.cdf(x, pos_mu-pos_sd).mean(axis=1)
# s_up = sp.stats.poisson.cdf(x, pos_mu+pos_sd).mean(axis=1)
# x = np.arange(30, step=0.1)

# fig, ax = plt.subplots()
# sns.ecdfplot(data, ax=ax, color='k',  label="Observed")
# ax.plot(x, mme, color='orangered', linestyle="--", label="μ mean")
# ax.fill_between(x, s_low, s_up, color='orangered', alpha=0.2, label="±SD mean")
# plt.legend()
# ax.set_axisbelow(True)
# ax.grid(alpha=0.2)
# ax.set_ylabel("Probability")
# ax.set_xlabel("Incubation Days")
# ax.set_title("Posterior CDF")
# plt.tight_layout()
# plt.savefig("sim_data_model_posterior.png", dpi=600)
# plt.show()
# plt.close()



# Fit simple Poisson model to sub-datasetes, one per site
summas = []
idatas = []
for k in range(len(datas)):
    with pm.Model() as mod:
        lamz = pm.Normal('lamz', mu=0, sigma=2)
        lam = pm.Normal('lam', mu=0, sigma=2)
        sigma = pm.TruncatedNormal('sigma', mu=0, sigma=2, lower=0)
        mu = pm.Deterministic("mu", pm.math.exp(lam + lamz*sigma))
        alpha = pm.TruncatedNormal('alpha', mu=0, sigma=2, lower=0)
        y = pm.NegativeBinomial('y', mu=mu, alpha=alpha, observed=datas[k])
        idata = pm.sample(2000, tune=2000, chains=4, nuts_sampler='numpyro', target_accept=0.95, random_seed=27)
    idatas.append(idata)
    
    summ = az.summary(idata, hdi_prob=0.9)
    summ['site'] = np.repeat(k, len(summ))
    summas.append(summ)
summs = pd.concat(summas)
summs.to_csv("summaries_sim_sites.csv")

fig, ax = plt.subplots(3,4, figsize=(20,10))
for k in range(len(datas)):
    idata = idatas[k]
    
    pos_mu = az.extract(idata)['mu'].values#.mean(axis=0)
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
    
    if k < 4:
        i=0
        j=k
    if k > 3  and k < 8:
        i=1
        j=k-4
    if k > 7:
        i=2
        j=k-8
        
    sns.ecdfplot(data, ax=ax[i,j], color='k',  label="Observed total")
    sns.ecdfplot(datas[k], ax=ax[i,j], color='dodgerblue', linestyle=":", label="Observed subset")
    ax[i,j].plot(x2, cdf_mean, color='orangered', linestyle="--", label="μ mean")
    ax[i,j].fill_between(x2, h5, h95, color='orangered', alpha=0.2, label="90% HDI")
    ax[i,j].legend()
    ax[i,j].set_axisbelow(True)
    ax[i,j].grid(alpha=0.2)
    ax[i,j].set_ylabel("Probability")
    ax[i,j].set_xlabel("Incubation Days")
    ax[i,j].set_title("Posterior CDF Site"+str(k+1))
plt.tight_layout()
plt.savefig("sim_data_posterior_sites.png", dpi=600)
plt.show()
plt.close()


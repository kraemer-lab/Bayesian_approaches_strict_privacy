# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from datetime import datetime
import scipy as sp

np.random.seed(27)

df = pd.read_csv("./data/nCoV-IDD-traveler-data_adjusted.csv")

df = df.dropna(subset='SL')
df = df.dropna(subset='SR')
df = df.dropna(subset='EL')
df = df.dropna(subset='ER')

dests = df['COUNTRY.DEST'].unique()

dfs = [df[df['COUNTRY.DEST']==d] for d in dests]

# order datasets from largest no. of datapoints to smallest
len_dfs = [len(d) for d in dfs]
len_dfs2 = sorted(len_dfs.copy(), reverse=True)
idxs = []
for d in range(len(dfs)):
    idx = np.where(np.array(len_dfs)==len_dfs2[d])[0][0]
    idxs.append(idx)
dfs_sorted = [dfs[i] for i in idxs] 

df = pd.concat(dfs_sorted)

EL = df.EL.values #exposure left boundary
EL = np.array([e.split(" ")[0] for e in EL])
ER = df.ER.values #exposure right boundary
ER = np.array([e.split(" ")[0] for e in ER])
SL = df.SL.values #symptoms left boundary
SL = np.array([e.split(" ")[0] for e in SL])
SR = df.SR.values #exposure right boundary
SR = np.array([e.split(" ")[0] for e in SR])

er = []
sr = []
for i in range(len(ER)): 
    er.append(datetime.strptime(ER[i], '%d/%m/%Y'))
    sr.append(datetime.strptime(SR[i], '%d/%m/%Y'))
er = np.array(er)
sr = np.array(sr)        
IR = []
for i in range(len(ER)): 
    IR.append(sr[i] - er[i])    
IR = [i.days for i in IR]
IR = np.array(IR)
IR[IR<0] = 0
maxInc = IR 

er = []
sl = []
for i in range(len(ER)): 
    er.append(datetime.strptime(ER[i], '%d/%m/%Y'))
    sl.append(datetime.strptime(SL[i], '%d/%m/%Y'))
er = np.array(er)
sl = np.array(sl)        
IL = []
for i in range(len(ER)): 
    IL.append(sl[i] - er[i])    
IL = [i.days for i in IL]
IL = np.array(IL)
IL[IL<0] = 0
minInc = IL 

right_upper = maxInc
right_lower = minInc

el = []
sr = []
for i in range(len(EL)): 
    el.append(datetime.strptime(EL[i], '%d/%m/%Y'))
    sr.append(datetime.strptime(SR[i], '%d/%m/%Y'))
el = np.array(el)
sr = np.array(sr)        
IR = []
for i in range(len(EL)): 
    IR.append(sr[i] - el[i])    
IR = [i.days for i in IR]
IR = np.array(IR)
IR[IR<0] = 0
maxInc = IR 

el = []
sl = []
for i in range(len(EL)): 
    el.append(datetime.strptime(EL[i], '%d/%m/%Y'))
    sl.append(datetime.strptime(SL[i], '%d/%m/%Y'))
el = np.array(el)
sl = np.array(sl)        
IL = []
for i in range(len(EL)): 
    IL.append(sl[i] - el[i])    
IL = [i.days for i in IL]
IL = np.array(IL)
IL[IL<0] = 0
minInc = IL 

left_upper = maxInc
left_lower = minInc

## EL and ER cannot be greater than SL or SR
left_upper[left_upper==0] = 1
left_lower[left_lower==0] = 1

right_upper[right_upper==0] = 1
right_lower[right_lower==0] = 1


left_interval1 = np.array([[x,y] for x,y in zip(left_upper, right_lower) if x!=y])
left_exact1 = np.array([[x] for x,y in zip(left_upper, right_lower) if x==y])

right_interval1 = np.array([[x,y] for x,y in zip(left_lower, right_upper) if x!=y])
right_exact1 = np.array([[x] for x,y in zip(left_lower, right_upper) if x==y])

left_interval2 = np.array([[x,y] for x,y in zip(left_lower, right_lower) if x!=y])
left_exact2 = np.array([[x] for x,y in zip(left_lower, right_lower) if x==y])

right_interval2 = np.array([[x,y] for x,y in zip(left_upper, right_upper) if x!=y])
right_exact2 = np.array([[x] for x,y in zip(left_upper, right_upper) if x==y])



site_idx = []
for s in range(len(dfs_sorted)):
    if s < 10:
        site = "S0"+str(s+1)
    else:
        site = "S"+str(s+1)
    site_idx.append(np.repeat(site, len(dfs_sorted[s])))        
site_idx = np.concatenate(site_idx)


data = pd.DataFrame({'SiteIdx':site_idx, 'Site':df['COUNTRY.DEST'].values, 
                      'left_lower':left_lower, 'left_upper':left_upper,
                      'right_lower':right_lower, 'right_upper':right_upper})

data.to_csv("./data/period_data.csv", index=False)

for s in data.SiteIdx:
    df = data[data.SiteIdx==s]
    df.to_csv("./chunked_data/df_"+s+".csv", index=False)

#####################################
########## Gamma model ############
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
    wl2 = pm.Potential('wl2', censored('censoredl2', alpha, beta, 
                                               left_interval2[:,1],
                                               left_interval2[:,0]))
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    yl2 = pm.Gamma("yl2", alpha, beta, observed=left_exact2[:,0])
    
    # latent likelihood of 'inexact' incubation periods
    wr2 = pm.Potential('wr2', censored('censoredr2', alpha, beta, 
                                               right_interval2[:,1],
                                               right_interval2[:,0]))
    # likelihood of exact incubation periods, i.e. lower = upper (minInc = mxInc)
    yr2 = pm.Gamma("yr2", alpha, beta, observed=right_exact2[:,0])
    
 
    ppc = pm.sample_prior_predictive(1000, random_seed=27)
    
mu_pri = az.extract(ppc.prior)['mu'].values
alp_pri = az.extract(ppc.prior)['alpha'].values
bet_pri = az.extract(ppc.prior)['beta'].values
plt.hist(mu_pri.T, bins=100)

with mod:
    idata = pm.sample(2000, tune=2000, nuts_sampler='numpyro', random_seed=27)


az.plot_trace(idata, kind='rank_vlines')

summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("gamma_summary_direct.csv")

pos_mu = az.extract(idata.posterior)['mu'].values
pos_sig = az.extract(idata.posterior)['sigma'].values
x = np.array([np.arange(30, step=0.1) for i in range(pos_mu.shape[0])]).T #30 days
cdf = sp.stats.gamma.cdf(x, pos_mu**2 / pos_sig**2)
cdf_m = cdf.mean(axis=1)
x2 = np.arange(30, step=0.1)  #30 days
plt.plot(x2, cdf_m)
plt.show()


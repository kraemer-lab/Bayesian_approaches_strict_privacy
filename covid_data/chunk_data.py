# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime

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


site_idx = []
for s in range(len(dfs_sorted)):
    if s < 9:
        site = "S0"+str(s+1)
    else:
        site = "S"+str(s+1)
    site_idx.append(np.repeat(site, len(dfs_sorted[s])))        
site_idx = np.concatenate(site_idx)


data = pd.DataFrame({'SiteIdx':site_idx, 'Site':df['COUNTRY.DEST'].values, 
                      'left_lower':left_lower, 'left_upper':left_upper,
                      'right_lower':right_lower, 'right_upper':right_upper})


for s in data.SiteIdx:
    df = data[data.SiteIdx==s]
    df.to_csv("./chunked_data/df_"+s+".csv", index=False)

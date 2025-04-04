import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma


#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

# Set seed for reproducibility
np.random.seed(27)

# Parameters
num_sites = 6
min_patients, max_patients = 18, 42
gamma_mean, gamma_sd = 8, 3
exact_prob = 0.1  # Only 10% exact observations

# Gamma parameters
shape = (gamma_mean ** 2) / (gamma_sd ** 2)
scale = (gamma_sd ** 2) / gamma_mean
rate = gamma_mean / gamma_sd**2

# Generate data
data = []
for site in range(1, num_sites + 1):
    n_patients = np.random.randint(min_patients, max_patients + 1)
    T_true = np.random.gamma(shape, scale, n_patients)
    
    is_exact = np.random.choice([True, False], size=n_patients, p=[exact_prob, 1-exact_prob])
    
    # For exact observations: lower = upper = true_time
    L = np.where(is_exact, T_true, np.nan)
    R = np.where(is_exact, T_true, np.nan)
    
    # For censored cases
    interval_idx = np.where(~is_exact)[0]
    L[interval_idx] = T_true[interval_idx] - np.random.uniform(1, 5, len(interval_idx))
    R[interval_idx] = T_true[interval_idx] + np.random.uniform(1, 5, len(interval_idx))
    
    # Ensure intervals are valid
    L = np.maximum(L, 0)
    R = np.maximum(R, L + 0.1)  # Force minimum interval width
    
    data.append(pd.DataFrame({
        'site_id': f"Site {site}",
        'true_time': T_true.round(0),
        'status': np.where(is_exact, 'exact', 'censored'),
        'min_incubation': L.round(0) + 1,
        'max_incubation': R.round(0) + 1,
    }))

df = pd.concat(data)

#### Plot data

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Forest Plot (Original Style) ---
site_means = df.groupby('site_id')['true_time'].mean().reset_index()

overall_mean = np.round(df['true_time'].mean(), 2)
overall_mode = np.round((shape - 1) / rate, 2)
overall_median = np.round(gamma.median(shape, scale=scale), 2)

# Original pointplot style you liked
ax1.errorbar(x=site_means['true_time'], y=range(len(site_means)), 
             xerr=df.groupby('site_id')['true_time'].sem(), lw=2,
             fmt='none', c='black', capsize=4, label="±SEM")
ax1.axvline(overall_mean, color='red', linestyle='--', lw=2, label=f'Overall Mean')
sns.pointplot(x='true_time', y='site_id', data=site_means, ax=ax1,
              join=False, color="steelblue", markers='o', scale=1.5, label="Site Mean")
ax1.set_title(r"$\bf{A.}$ Average Incubation Period per Site", fontsize=18, loc="left")
ax1.set_xlabel('Time (Days)', fontsize=16)
ax1.set_ylabel('Site', fontsize=16)
ax1.legend(fontsize=14)
ax1.grid(axis='x', alpha=0.3)

# --- Plot 2: Gamma Distribution (Original Style) ---
x = np.linspace(0, 20, 1000)
pdf = gamma.pdf(x, a=shape, scale=scale)
sns.histplot(df['true_time'], ax=ax2, kde=False, stat='density', 
             alpha=0.5, label='Observed Data')
ax2.plot(x, pdf, 'r-', lw=3, label=f'Gamma(μ={gamma_mean}, σ={gamma_sd})')
ax2.axvline(overall_mean, color='k', linestyle='--', lw=2, label=f"Mean={overall_mean}")
ax2.axvline(overall_median, color='k', linestyle=':', lw=2, label=f"Median={overall_median}")
ax2.axvline(overall_mode, color='k', linestyle='-.', lw=2, label=f"Mode={overall_mode}")
ax2.set_title(r"$\bf{B.}$ Distribution Pooled Across Sites", fontsize=18, loc="left")
ax2.set_xlabel('Time (Days)', fontsize=16)
ax2.set_ylabel('Density', fontsize=16)
ax2.legend(fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("simulations_summary.png", dpi=600)
plt.show()


# df.to_csv("df_all_data.csv", index=False)

sites = df.site_id.unique()
for i in range(len(sites)):
    d = df[df.site_id==sites[i]] 
    d.reset_index(inplace=True, drop=True)
    d.to_csv("./chunked_data/site_"+str(i+1)+".csv")


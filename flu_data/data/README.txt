This README and csv data are from: https://datadryad.org/stash/dataset/doi:10.5061/dryad.g2j62
Part of the study referred below.

Data and R syntax for the following publication:
Victor Virlogeux, Juan Yang, Vicky J. Fang, Luzhao Feng, Tim K. Tsang, Hui Jiang, Peng Wu, Jiandong Zheng, Eric H. Y. Lau, Ying Qin, Zhibin Peng, J. S. Malik Peiris, Hongjie Yu, Benjamin J. Cowling. Association between the severity of influenza A(H7N9) virus infections and length of the incubation period. PLoS ONE 2015.


+-----------------+
|  List of files  |
+-----------------+

1. [data_h7n9_severity.csv] This data file contains information on 395 human cases of laboratory-confirmed H7N9 infection in China between Mar 2013 and Aug 2014.
Definition of column headings >>>
Case No.: case identifier number.
IncP_min/IncP_Max: the lower & upper bound of the incubation, i.e., delay (in days) from exposure to symptom onset.
age: age of each case, in years.
sex_status: 0- female; 1- male.
missing_exposure: 0- exposure dates given; 1- exposure dates missing.
death_status: 0- alive; 1- dead.
non_capital: 0- either capital city or rural area; 1- non-capital city.
rural: 0- urban area; 1- rural area.
underlying_condition: 0- without any underlying condition; 1- with one or more underlying conditions.

2. [mcmc_function_incubation.r] R syntax to define the MCMC functions to be used in other scripts.

3. [Table_1.r] R syntax to reproduce results in Table 1.

4. [Table_2.r] R syntax to reproduce results in Table 2.

5. [Table_3.r] R syntax to reproduce results in Table 3.

6. [Table_4.r] R syntax to reproduce results in Table 4.

7. [Figure_1.r] R syntax to reproduce Figure 1.

8. [Figure_2.r] R syntax to reproduce Figure 2.

Note: The numerical results reported in the main text of the paper are extracted from the respective tables, apart from the reported incubation periods of fatal and non-fatal cases as well as the difference between them. The syntax to generate the above mentioned incubation periods (as well as 95% CI) is provided at the middle (before step 2) of the script [Table_2.r].


 












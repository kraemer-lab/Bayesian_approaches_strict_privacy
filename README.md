<h1> Bayesian  Approaches to Collaborative Data Analysis with Strict Privacy Restrictions </h1>

<p> In this repository you will find code and data to reproduce results from the Bayesian  Approaches to Collaborative Data Analysis with Strict Privacy Restrictions paper. </p>

<p> The <b>simulations</b> directory contains code for simulating and analysing data for the Bayesian statistical approaches applied to simulated data. </p>

<p> The <b>flu_data</b> directory contains data and code for the Bayesian statistical approaches applied to Avian Influenza A (H7H9) incubation period data (Virlogeux et al., 2016). </p>

<p> The <b>covid_data</b> directory contains data and code for the Bayesian statistical approaches applied to Corona Virus Disease 2019 (COVID-19) incubation period data (Lauer et al., 2020). </p>

<p> Analyses can be carried on in any statistical software supporting HMC sampling for Bayesian models, such as Stan or PyMC. Current analyses are carried on PyMC (Abril-Pla et al., 2023): https://www.pymc.io/welcome.html.

All required software packages are in the .py files in each of the above directories. We recommend the following setup:

```
conda create -n pymc_env
conda activate pymc_env
conda install -c conda-forge pymc
conda install m2w64-toolchain
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
pip install numpyro
pip install git+https://github.com/pymc-devs/pymc-experimental.git
```

</p>


<h2> References </h2>

<p>

Abril-Pla, O., Andreani, V., Carroll, C., Dong, L. Y., Fonnesbeck, C., Kochurov, M., Kumar, R., Lao, J., Luhmann, C. C., Martin, O. A., Osthege, M., Vieira, R., Wiecki, T. V., & Zinkov, R. (2023). PyMC: a modern, and comprehensive probabilistic programming framework in Python. PeerJ, 9, e1516–e1516. https://doi.org/10.7717/peerj-cs.1516

Lauer, S. A., Grantz, K. H., Bi, Q., Jones, F. K., Zheng, Q., Meredith, H. R., Azman, A. S., Reich, N. G., & Lessler, J. (2020). The Incubation Period of Coronavirus Disease 2019 (COVID-19) From Publicly Reported Confirmed Cases: Estimation and Application. Annals of Internal Medicine, 172(9), 577–582. https://doi.org/10.7326/M20-0504

Virlogeux, V., Yang, J., Fang, V. J., Feng, L., Tsang, T. K., Jiang, H., Wu, P., Zheng, J., Lau, E. H. Y., Qin, Y., Peng, Z., Peiris, J. S. M., Yu, H., & Cowling, B. J. (2016). Association between the Severity of Influenza A(H7N9) Virus Infections and Length of the Incubation Period. PLOS ONE, 11(2), e0148506. https://doi.org/10.1371/journal.pone.0148506
</p>


<h3>MIT License</h3>

<p>
Copyright (c) 2024 Kraemer Lab, University of Oxford.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
</p>

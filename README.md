# Group_Variable_Importance
### Conda environment
The required packages are installed as a new *conda* environment including both R and Python dependencies with the following command:

```
conda env create -f requirements_conda.yml
```
> :warning: The use of ```mamba``` is faster and more stable for packages installation.

### R environment
The missing R packages can be found in the "requirements_r.rda" file and can be
downloaded using the following commands:

```
R

load("requirements_r.rda")

for (count in 1:length(installedpackages)) {
    install.packages(installedpackages[count])
}
```
> :warning: For ```reticulate```, if asked for default python virtual environment, the answer should be ```no``` to take the default *conda* environment into consideration

## For the **First experiment**:
### Results computation with R script (```compute_simulations```):
  * Set ```DEBUG``` to ```FALSE```.
  * ```N_SIMULATIONS``` is set to the range (`1`, `100`)
  * With ```N_CPU``` > 1, the parallel processing is used
  * The list of methods contains (```marginal```, ```permfit```,
    ```cpi```, ```cpi_rf```, ```gpfi```, ```gopfi```, ```dgi``` and
    ```goi```).
  * ```n_samples``` is set to `1000` and ```n_featues``` is set to `50` 
  * ```rho_group``` lists all the correlation strengths in this experiment (`0`, `0.2`,
    `0.5`, `0.8`)
  * Number of permutations/samples ```n_perm``` is set to `100`
  * The output *csv* file is found in ```results/results_csv```

### Results plotting:
  * Preparing *csv* files with R script ```plot_simulations_all``` under ```[AUC-type1error-power-time_bars]_blocks_100_grps.csv```
  * The plotting is done under ```plots/plot_figure_simulations_grps.ipynb``` with:
    * `Figure 1` for the Figure 2 in the main text
    * `Power + Time + Prediction scores` for the Figure 6 in the supplement
    * `Figure 1 Calibration` for the Figure 5 in the supplement

## For the **Second experiment**:
  * We use ``` compute_simulations_groups```.
  * The script can be launched with the following command:
    ```
    python -u compute_simulations_groups.py --n 1000 --pgrp 100 --nblocks 10 --intra 0.8 --inter 0.8 --conditional 1 --stacking 1 --f 1 --s 100 --njobs 1
    ```
    * ```--n``` stands for the number of samples (Default `1000`)
    * ```--pgrp``` stands for number of variables per group (Default `100`)
    * ```--nblocks``` stands for the number of blocks/groups in the data
      structure (Default `10`)
    * ```--intra``` stands for the intra correlation inside the groups
      (Default `0.8`)
    * ```--inter``` stands for the inter correlation between the groups
      (Default `0.8`)
    * ```--conditional``` stands for the use of CPI (`1`) or PI (`0`)
    * ```--stacking``` stands for the use of stacking (`1`) or not (`0`)
    * ```--f``` stands for the first point of the range (Default `1`)
    * ```--s``` stands for the step-size i.e. range size (Default `100`)
    * ```--njobs``` stands for the serial/parallel implementation under
      `Joblib` (Default `1`)
  * The output csv file is found in ```results/results_csv``` under ```[AUC-type1error-power-time_bars]_blocks_100_groups_CPI_n_1000_p_1000_1::100_folds_2.csv```
  * The plotting is done under ```plots/plot_figure_simulations_grps.ipynb``` with
    `Compare Stacking vs Non Stacking` for the Figure 3 in the main text

## For the **Third experiment**:
  ### Scoring results 10-fold cross validated with ```process_var_groups_outer```:
  * The data are the public data from `UKBB` that needs to sign an agreement before using it (Any personal data are already removed)
  * The ```biomarker``` is set by default to `age`
  * ```n_jobs``` stands for serial/parallel computations
  * ```k_fold_bbi``` stands for the number of folds for the internal cross validation
    of the method
  * ```k_fold``` stands for the number of folds for train/test splitting the
    original data
  
  ### Significance & Performance 10-fold cross validated with ```process_var_groups_outer_post```:
  * The $\underline{representative}$ `p-value` will be 2*median(p-values) across the 10 folds
  * As for the $\underline{performance}$, it is measured on the 10% test set
    split per fold
  * The output *csv* file is found in ```results/results_csv``` under
    ```Result_UKBB_age_all_imp_10_outer_2_inner_PERF.csv``` and ```Result_UKBB_age_all_imp_10_outer_2_inner_SIGN.csv```
  * The plotting is done under ```plots/plot_figure_simulations_grps.ipynb``` with
    `Figure 3` for the Figure 4 in the main text
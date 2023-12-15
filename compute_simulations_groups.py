import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from permfit_python.BBI_pytorch import BlockBasedImportance
import time
from scipy.stats import norm
from scipy.linalg import cholesky
import pickle


parser = argparse.ArgumentParser()
# Number of samples
parser.add_argument("--n", type=int)
# Variables per group
parser.add_argument("--pgrp", type=int)
# Number of blocks
parser.add_argument("--nblocks", type=int)
# Intra Correlation
parser.add_argument("--intra", type=float)
# Inter Correlation
parser.add_argument("--inter", type=float)
# CPI or PI
parser.add_argument("--conditional", type=int)
# Stacking or not
parser.add_argument("--stacking", type=int)
# Starting iteration
parser.add_argument("--f", type=int)
# Stepsize
parser.add_argument("--s", type=int)
# Number of jobs
parser.add_argument("--njobs", type=int)
args, _ = parser.parse_known_args()


def generate_cor_blocks(
    p, inter_cor, intra_cor, n_blocks, vars_per_grp=None
):
    if vars_per_grp is None:
        vars_per_grp = [int(p / n_blocks)] * n_blocks
    curr_vars_per_grp = vars_per_grp.copy()
    curr_vars_per_grp.insert(0, 0)
    curr_vars_per_grp = np.cumsum(curr_vars_per_grp)
    cor_mat = np.zeros((p, p))
    cor_mat.fill(inter_cor)
    for i in range(n_blocks):
        cor_mat[
            curr_vars_per_grp[i] : curr_vars_per_grp[i + 1],
            curr_vars_per_grp[i] : curr_vars_per_grp[i + 1],
        ] = intra_cor
    np.fill_diagonal(cor_mat, 1)
    return cor_mat


def generate_blocks(p, n_blocks, vars_per_grp=None):
    list_grps = []
    if vars_per_grp is None:
        vars_per_grp = [int(p / n_blocks)] * n_blocks
    curr_vars_per_grp = vars_per_grp.copy()
    curr_vars_per_grp.insert(0, 0)
    curr_vars_per_grp = np.cumsum(curr_vars_per_grp)
    for i in range(n_blocks):
        list_grps.append(
            [
                str(el)
                for el in np.arange(
                    curr_vars_per_grp[i], curr_vars_per_grp[i + 1]
                )
            ]
        )
    return list_grps


def compute_simulations(
    seed,
    filename=None,
    resample=False,
    list_nominal=None,
    snr=4,
    conditional=False,
    group_stacking=False,
    n=1000,
    p=50,
    inter_cor=0.8,
    intra_cor=0,
    n_blocks=10,
    vars_per_grp=None,
    prob_type="regression",
):
    start = time.time()
    rng = np.random.RandomState(seed)
    # Initializing the groups

    if filename is not None:
        if resample:
            data = pd.read_csv(f"{filename}.csv")
            with open("groups_UKBB.pickle", "rb") as file:
                groups = pickle.load(file)

        if isinstance(groups, dict):
            list_grps = [groups[el] for el in groups.keys()]

        new_grps = [[] for _ in range(len(groups))]
        for ind_grp, grp in enumerate(groups):
            for el in groups[grp]:
                if el not in (
                    list_nominal["nominal"]
                    + list_nominal["ordinal"]
                    + list_nominal["binary"]
                ):
                    new_grps[ind_grp].append(el)

        new_grps = [el for el in new_grps if len(el) != 0]
        list_grps = new_grps.copy()
        data = data.loc[
            :,
            [
                el
                for el in data.columns
                if el
                not in (
                    list_nominal["nominal"]
                    + list_nominal["ordinal"]
                    + list_nominal["binary"]
                )
            ],
        ]
        list_nominal = {"nominal": [], "ordinal": [], "binary": []}

    else:
        list_grps = generate_blocks(p, n_blocks, vars_per_grp)
        cor_mat = generate_cor_blocks(
            p, inter_cor, intra_cor, n_blocks, vars_per_grp
        )
        x = norm.rvs(size=(p, n), random_state=seed)
        c = cholesky(cor_mat, lower=True)
        data = pd.DataFrame(
            np.dot(c, x).T, columns=[str(i) for i in np.arange(p)]
        )

    data_enc = data.copy()
    list_cols = np.array(
        [
            np.array(el)[np.arange(max(1, int(0.1 * len(el))))]
            for el in list_grps[:5]
        ]
    )
    data_enc_a = data_enc.loc[:, np.concatenate(list_cols, axis=0)]

    enc_dict = {}
    total_labels_enc = []
    if len(list_nominal["nominal"]) > 0:
        for col_encode in list_nominal["nominal"]:
            if col_encode in data_enc_a.columns:
                enc = OneHotEncoder(handle_unknown="ignore")
                enc.fit(data_enc_a[[col_encode]])
                labeled_cols = [
                    enc.feature_names_in_[0]
                    + "_"
                    + str(enc.categories_[0][j])
                    for j in range(len(enc.categories_[0]))
                ]
                total_labels_enc += labeled_cols
                hot_cols = pd.DataFrame(
                    enc.transform(data_enc_a[[col_encode]]).toarray(),
                    dtype="int32",
                    columns=labeled_cols,
                )
                data_enc_a = data_enc_a.drop(columns=[col_encode])
                data_enc_a = pd.concat([data_enc_a, hot_cols], axis=1)
                enc_dict[col_encode] = enc

    count_pairs = 0
    tmp_comb = data_enc_a.shape[1]

    # Determine beta coefficients
    effectset = [-0.5, -1, -2, -3, 0.5, 1, 2, 3]
    beta = rng.choice(
        effectset, size=(tmp_comb + count_pairs), replace=True
    )

    # Generate response
    ## The product of the signal predictors with the beta coefficients
    prod_signal = np.dot(data_enc_a, beta)

    if prob_type != "classification":
        sigma_noise = np.linalg.norm(prod_signal, ord=2) / (
            snr * np.sqrt(data_enc_a.shape[0])
        )
        y = prod_signal + sigma_noise * rng.normal(
            size=prod_signal.shape[0]
        )
    else:
        from scipy.special import expit

        y = rng.binomial(
            1, size=data_enc_a.shape[0], p=expit(prod_signal)
        ).astype(str)
        while y.tolist().count("0") < 0.1 * data_enc_a.shape[0]:
            y = rng.binomial(
                1, size=data_enc_a.shape[0], p=expit(prod_signal)
            ).astype(str)

    # Initializing the model
    bbi_model = BlockBasedImportance(
        estimator=None,
        do_hyper=True,
        importance_estimator="Mod_RF",
        dict_hyper=None,
        conditional=conditional,
        n_perm=100,
        prob_type=prob_type,
        k_fold=2,
        list_nominal=list_nominal,
        groups=list_grps,
        group_stacking=group_stacking,
        verbose=10,
        n_jobs=1,
    )
    bbi_model.fit(data_enc, y)
    res = bbi_model.compute_importance()

    elapsed = time.time() - start
    method = "Permfit-DNN" if not conditional else "CPI-DNN"

    f_res = {}
    f_res["method"] = [method] * len(list_grps)
    f_res["score"] = res["score_R2"]
    f_res["elapsed"] = [elapsed] * len(list_grps)
    f_res["correlation"] = [intra_cor] * len(list_grps)
    f_res["correlation_group"] = [inter_cor] * len(list_grps)
    f_res["n_samples"] = [data.shape[0]] * len(list_grps)
    f_res["prob_data"] = ["regression"] * len(list_grps)
    f_res["iteration"] = seed
    f_res["group_stack"] = [group_stacking] * len(list_grps)

    f_res = pd.DataFrame(f_res)

    # Add importance, std & p-val
    df_imp = pd.DataFrame(res["importance"], columns=["importance"])
    df_pval = pd.DataFrame(res["pval"], columns=["p_value"])

    f_res = pd.concat([f_res, df_imp, df_pval], axis=1)
    return f_res


filename = None
list_nominal = {"nominal": [], "ordinal": [], "binary": []}

# Configuration
vars_per_grp = [args.pgrp] * args.nblocks
group_stacking = True if args.stacking == 1 else False
conditional = True if args.conditional == 1 else False
n = args.n
p = sum(vars_per_grp) if vars_per_grp is not None else 500
f_d = args.f
l_d = args.f + args.s

print(f"Range: {f_d} to {l_d-1}")
parallel = Parallel(n_jobs=args.njobs, verbose=1)
final_res = parallel(
    delayed(compute_simulations)(
        seed=seed,
        filename=filename,
        resample=False,
        list_nominal=list_nominal,
        snr=4,
        conditional=conditional,
        group_stacking=group_stacking,
        n=n,
        p=p,
        inter_cor=args.inter,
        intra_cor=args.intra,
        n_blocks=len(vars_per_grp) if vars_per_grp is not None else 10,
        vars_per_grp=vars_per_grp,
        prob_type="regression",
    )
    for seed in np.arange(f_d, l_d)
)

final_res = pd.concat(final_res).reset_index(drop=True)
final_res.to_csv(
    f"results/results_csv/simulation_results_blocks_100_groups_n_{n}_p_{p}_{f_d}::{l_d-1}_{'stack' if group_stacking else 'non_stack'}_{'cpi' if conditional else 'permfit'}.csv",
    index=False,
)

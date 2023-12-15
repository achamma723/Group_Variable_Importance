import collections
from itertools import chain
import pandas as pd
import sys
sys.path.insert(1, "..")

from sklearn.model_selection import KFold
from permfit_python.BBI_pytorch import BlockBasedImportance

from ukbb_variables import (
    brain_dmri_fa,
    brain_dmri_icvf,
    brain_dmri_isovf,
    brain_dmri_l1,
    brain_dmri_l2,
    brain_dmri_l3,
    brain_dmri_md,
    brain_dmri_mo,
    brain_dmri_od,
    brain_smri_plus,
    earlylife,
    brain_smri,
    education,
    lifestyle,
    mental_health,
    primary_demographics,
)


list_imp = [
    brain_dmri_fa,
    brain_dmri_icvf,
    brain_dmri_isovf,
    brain_dmri_l1,
    brain_dmri_l2,
    brain_dmri_l3,
    brain_dmri_md,
    brain_dmri_mo,
    brain_dmri_od,
    brain_smri_plus,
    earlylife,
    education,
    lifestyle,
    mental_health,
    primary_demographics,
]

list_imp_labels = [
    "brain_dmri_fa",
    "brain_dmri_icvf",
    "brain_dmri_isovf",
    "brain_dmri_l1",
    "brain_dmri_l2",
    "brain_dmri_l3",
    "brain_dmri_md",
    "brain_dmri_mo",
    "brain_dmri_od",
    "brain_smri_plus",
    "earlylife",
    "education",
    "lifestyle",
    "mental_health",
    "primary_demographics",
]

id_dict = collections.OrderedDict(
    chain(
        brain_dmri_fa.items(),
        brain_dmri_icvf.items(),
        brain_dmri_isovf.items(),
        brain_dmri_l1.items(),
        brain_dmri_l2.items(),
        brain_dmri_l3.items(),
        brain_dmri_md.items(),
        brain_dmri_mo.items(),
        brain_dmri_od.items(),
        brain_smri_plus.items(),
        earlylife.items(),
        education.items(),
        lifestyle.items(),
        mental_health.items(),
        brain_smri.items(),
        primary_demographics.items(),
    )
)

nominal_columns = list(
    pd.read_csv("ukbb_data_age_no_hot_encoding_nominal_columns.csv")["x"]
)
ordinal_columns = list(
    pd.read_csv("ukbb_data_age_no_hot_encoding_ordinal_columns.csv")["x"]
)
binary_columns = list(
    pd.read_csv("ukbb_data_age_no_hot_encoding_binary_columns.csv")["x"]
)

list_cat = {
    "nominal": nominal_columns,
    "ordinal": ordinal_columns,
    "binary": binary_columns,
}

dict_biomarker = {
    "age": "21022-0.0",
    "alcohol": "1558-2.0",
    "intelligence": "20016-2.0",
    "met": "22040-0.0",
    "neuroticism": "20127-0.0",
    "sleep": "1160-2.0",
    "smoke": "20161-2.0",
}


def process_var(biomarker, list_cat, k_fold, k_fold_bbi):
    col_pred = dict_biomarker[biomarker]
    df = pd.read_csv(f"ukbb_data_{biomarker}_no_hot_encoding.csv")

    # Remove tangent columns
    cols_tangent = [i for i in list(df.columns) if "-" not in i]
    # df = df.drop(columns=cols_tangent)

    grps = {"connectivity": cols_tangent}
    for imp in range(len(list_imp)):
        grps[list_imp_labels[imp]] = []
        for el in list(list_imp[imp]):
            if (el in df.columns) and (el != col_pred):
                grps[list_imp_labels[imp]].append(el)
        if len(grps[list_imp_labels[imp]]) == 0:
            grps.pop(list_imp_labels[imp], "None")

    # Merging brain sub-groups
    grps_merged = {"brain": []}
    for key in grps.keys():
        if key.split("_")[0] in ["brain", "connectivity"]:
            grps_merged["brain"] += grps[key]
        else:
            grps_merged[key] = grps[key]

    X = df.loc[:, df.columns != col_pred]
    y = df[col_pred]

    list_cat["nominal"] = [
        i for i in list_cat["nominal"] if i in list(X.columns)
    ]
    list_cat["ordinal"] = [
        i for i in list_cat["ordinal"] if i in list(X.columns)
    ]
    list_cat["binary"] = [
        i for i in list_cat["binary"] if i in list(X.columns)
    ]

    # Configuration
    conditional = True
    n_jobs = 1
    group_stacking = True
    n_perm = 100
    random_state = 2023
    groups = grps

    # Implementing k-fold cross validation as the default behavior
    kf = KFold(n_splits=k_fold, random_state=random_state, shuffle=True)
    list_model = []
    for ind_fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Processing outer fold: {ind_fold+1}")
        X_train, _ = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, _ = y[train_index], y[test_index]

        # Model initialization
        bbi_model = BlockBasedImportance(
            importance_estimator="Mod_RF",
            prob_type="regression",
            conditional=conditional,
            k_fold=k_fold_bbi,
            n_jobs=n_jobs,
            list_nominal=list_cat,
            group_stacking=group_stacking,
            groups=groups,
            random_state=random_state,
            n_perm=n_perm,
            verbose=10,
            com_imp=True,
        )

        bbi_model.fit(X_train.reset_index(drop=True), y_train)
        results = bbi_model.compute_importance()

        list_model.append(
            pd.DataFrame(
                {
                    "method": ["CPI-DNN"] * len(groups),
                    "variable": list(groups.keys()),
                    "importance": results["importance"][:, 0],
                    "p_value": results["pval"][:, 0],
                    "iteration": ind_fold + 1,
                }
            )
        )
    return list_model


biomarker = "age"
k_fold = 10
k_fold_bbi = 2

# for biomarker in biomarkers:
pd.concat(
    process_var(biomarker, list_cat, k_fold, k_fold_bbi), axis=0
).to_csv(
    f"../results/results_csv/Result_UKBB_{biomarker}_all_imp_{k_fold}_outer_{k_fold_bbi}_inner.csv",
    index=False,
)

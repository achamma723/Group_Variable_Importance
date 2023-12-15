sklearn <- import("sklearn", convert = FALSE)


compute_marginal <- function(sim_data,
                             prob_type = "regression",
                             list_grps = list(),
                             ...) {
    print("Applying Marginal Method")

    # Splitting train/test sets
    train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
    marginal_imp <- numeric()
    marginal_pval <- numeric()
    score_val <- 0
    if (length(list_grps) == 0)
        indices = paste0("x", 1:ncol(sim_data[, -1]))
    else
        indices = list_grps

    count_ind = 1
    if (prob_type == "classification") {
        sim_data$y <- as.factor(sim_data$y)
        for (i in indices) {
            i = paste0(i, collapse="+")
            fit <- glm(formula(paste0("y ~ ", i)),
                data = sim_data[train_ind, ],
                family = binomial()
            )
            sum_fit <- summary(fit)
            marginal_imp[count_ind] <- coef(sum_fit)[, 1][[2]]
            marginal_pval[count_ind] <- coef(sum_fit)[, 4][[2]]
            pred <- predict(fit, newdata = sim_data[-train_ind, -1], type="response")
            score_val <- score_val +
                py_to_r(sklearn$metrics$roc_auc_score(sim_data$y[-train_ind], pred))
            count_ind <- count_ind + 1
          }
      }

    if (prob_type == "regression") {
        sim_data$y <- as.numeric(sim_data$y)
        for (i in indices) {
            i = paste0(i, collapse="+")
            fit <- glm(formula(paste0("y ~ ", i)),
                data = sim_data[train_ind, ]
            )
            sum_fit <- summary(fit)
            marginal_imp[count_ind] <- coef(sum_fit)[, 1][[2]]
            marginal_pval[count_ind] <- coef(sum_fit)[, 4][[2]]
            pred <- predict(fit, newdata = sim_data[-train_ind, -1])
            score_val <- score_val + py_to_r(sklearn$metrics$r2_score(sim_data$y[-train_ind], pred))
            count_ind <- count_ind + 1
        }
      }

    return(data.frame(
        method = "Marg",
        importance = marginal_imp,
        p_value = marginal_pval,
        score = score_val / ncol(sim_data[, -1])
    ))
}


compute_permfit <- function(sim_data,
                           index_i,
                           n = 1000L,
                           prob_type = "regression",
                           n_perm = 100,
                           n_jobs = 1,
                           backend = "loky",
                           nominal = NULL,
                           list_grps = list(),
                           group_stack = FALSE,
                           ...) {
    print("Applying DNN Permfit Method")

    bbi <- import_from_path("BBI_pytorch",
        path = "permfit_python"
    )

    bbi_model <- bbi$BlockBasedImportance(
        prob_type = prob_type,
        index_i = index_i,
        conditional = FALSE,
        k_fold = 2L,
        n_perm = n_perm,
        n_jobs = n_jobs,
        list_nominal = nominal,
        groups = list_grps,
        group_stacking = group_stack
    )

    bbi_model$fit(
        sim_data[, -1],
        as.matrix(sim_data$y)
    )

    results <- bbi_model$compute_importance()

    return(data.frame(
        method = "Permfit-DNN",
        importance = results$importance,
        p_value = results$pval,
        score = results$score_R2
    ))
}


compute_cpi <- function(sim_data,
                                index_i,
                                n = 1000L,
                                prob_type = "regression",
                                depth = 2,
                                n_perm = 100,
                                n_jobs = 1,
                                backend = NULL,
                                perm = FALSE,
                                nominal = NULL,
                                list_grps = list(),
                                group_stack = FALSE,
                                ...) {
    print("Applying DNN Conditional Method")

    bbi <- import_from_path("BBI_pytorch",
        path = "permfit_python"
    )

    bbi_model <- bbi$BlockBasedImportance(
        importance_estimator = 'Mod_RF',
        prob_type = prob_type,
        index_i = index_i,
        conditional = TRUE,
        k_fold = 2L,
        n_perm = n_perm,
        n_jobs = n_jobs,
        list_nominal = nominal,
        groups = list_grps,
        group_stacking = group_stack,
        Perm = perm
    )

    bbi_model$fit(
        sim_data[, -1],
        as.matrix(sim_data$y)
    )

    results <- bbi_model$compute_importance()

    return(data.frame(
        method = "CPI-DNN",
        importance = results$importance[, 1],
        p_value = results$pval[, 1],
        score = results$score_R2
    ))
}


compute_cpi_rf <- function(sim_data,
                                index_i,
                                n = 1000L,
                                prob_type = "regression",
                                depth = 2,
                                n_perm = 100,
                                n_jobs = 1,
                                backend = NULL,
                                perm = FALSE,
                                nominal = NULL,
                                list_grps = list(),
                                group_stack = FALSE,
                                ...) {
    print("Applying DNN Conditional with RF")

    bbi <- import_from_path("BBI_pytorch",
        path = "permfit_python"
    )

    bbi_model <- bbi$BlockBasedImportance(
        estimator = 'RF',
        # importance_estimator = 'Mod_RF',
        prob_type = prob_type,
        index_i = index_i,
        conditional = TRUE,
        k_fold = 2L,
        n_perm = n_perm,
        n_jobs = n_jobs,
        list_nominal = nominal,
        groups = list_grps,
        group_stacking = group_stack,
        Perm = perm
    )

    bbi_model$fit(
        sim_data[, -1],
        as.matrix(sim_data$y)
    )

    results <- bbi_model$compute_importance()

    return(data.frame(
        method = "CPI-RF",
        importance = results$importance[, 1],
        p_value = results$pval[, 1],
        score = results$score_R2
    ))
}


compute_grps <- function(sim_data,
                         prob_type = "regression",
                         num_trees = 2000L, 
                         list_grps = list(),
                         func = "gpfi",
                         ...) {
    task <- makeRegrTask(data = sim_data, target = "y")
    # RF model
    learner <- makeLearner("regr.ranger", par.vals = list(num.trees = num_trees))
    mod <- train(learner = learner, task = task)
    if (prob_type == "regression")
        res <- resample(learner, task, cv5, measures = mse, models = TRUE)
    gimp <- Gimp$new(task = task, res = res, mod = mod, lrn = learner)
    
    group <- c()
    for (i in 1:length(list_grps))
        group <- c(group, rep(paste0("G", i), length(list_grps[[i]])))
    group_df <- data.frame(feature = colnames(sim_data[, -1]), group = group, stringsAsFactors = FALSE)
    if (prob_type == "regression") {
        if (func == "gpfi") {
            res <- gimp$group_permutation_feat_imp(group_df, PIMP = FALSE, n.feat.perm = 100, regr.measure = mse)
            list_grps_new <- list()
            for (grp_ind in 1:length(list_grps)) {
                curr_grp <- ""
                for (i in list_grps[[grp_ind]])
                    curr_grp <- paste(c(curr_grp, i), collapse=",")
                list_grps_new[[substring(curr_grp, 2)]] <- grp_ind
            }

            for (i in 1:dim(res)[1]) {
                res$features[i] <- list_grps_new[[res$features[i]]]
            }
            res <- res[mixedorder(as.character(res$features)), ]
            imp <- res$mse
        }
        if (func == "gopfi") {
            res <- gimp$group_only_permutation_feat_imp(group_df, PIMP = FALSE, n.feat.perm = 100, regr.measure = mse)
            res <- res[mixedorder(as.character(res$group_id)), ]
            res <- res[-1, ]
            imp <- res$GOPFI
        }
        if (func == "dgi") {
            res_list <- gimp$drop_group_importance(group_df, measures = mse)
            res <- c()
            for (i in 1:length(list_grps)) {
                res <- c(res, res_list[[i]]$aggr - res_list$all$aggr)
            }
            imp <- res
        }
        if (func == "goi") {
            res_list <- gimp$group_only_importance(group_df, measures = mse)
            res <- c()
            for (i in 1:length(list_grps)) {
                res <- c(res, res_list$featureless$aggr - res_list[[i]]$aggr)
            }
            imp <- res
        }
    }
    return(data.frame(
        method = func,
        importance = imp,
        p_value = NA,
        score = NA
        ))
}

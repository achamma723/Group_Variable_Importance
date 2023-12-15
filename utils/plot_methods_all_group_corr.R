# load relevant R package
require(data.table)
suppressMessages({
  library("ROCR")
  library("ggplot2")
  library("dplyr")
  library("scales")
})


compute_auc <- function(obj,
                        ...) {

  if(is.na(obj$p_value[[1]])) {
  imp_vals <- as.numeric(obj$importance)
  }
  else {
  imp_vals <- as.numeric(-obj$p_value)
  }

  ground_tr <- rep(0, length(imp_vals))
  ground_tr[1] <- 0
  ground_tr[2] <- 1
  ground_tr[3] <- 1
  ground_tr[4] <- 1
  ground_tr[5] <- 1

  return(performance(
    prediction(
      imp_vals,
      ground_tr
    ),
    "auc"
  )@y.values[[1]])
}


compute_pval <- function(obj,
                         upper_bound = 0.05) {
  if (length(obj$p_value[!is.na(obj$p_value)]) == 0) {
    return(NULL)
  }
  return(mean(obj$p_value[-c(1, 2, 3, 4, 5)] < upper_bound))
}


compute_power <- function(obj,
                          upper_bound = 0.05) {
  if (length(obj$p_value[!is.na(obj$p_value)]) == 0) {
    return(NULL)
  }
return(mean(obj$p_value[c(1, 2, 3, 4, 5)] < upper_bound))
}


plot_time <- function(source_file,
                      output_file,
                      list_func = NULL,
                      N_CPU = 10) {
  df <- fread(source_file)

  res <- df[,
    mean(elapsed),
    by = .(
      n_samples,
      method,
      iteration,
      prob_data,
      group_stack
    )
  ]

  write.csv(res, file=file.path(
      "results/results_csv",
      paste0(
        output_file, ".csv"
      )
    ))
}


plot_method <- function(source_file,
                        output_file,
                        func = NULL,
                        upper_bound = 0.05,
                        title = "AUC",
                        list_func = NULL) {
  df <- fread(source_file)

  res <- df[,
    func(c(.BY, .SD),
      upper_bound = upper_bound
    ),
    by = .(
      method,
      correlation_group,
      n_samples,
      prob_data,
      iteration,
      group_stack
    )
  ]

  write.csv(res, file=file.path(
      "results/results_csv",
      paste0(
        output_file, ".csv"
      )
    ))
}
suppressMessages({
    source("utils/plot_methods_all_group_corr.R")
    library("tools")
    library("data.table")
    library("ggpubr")
    library("scales")
})

file_path <- paste0(getwd(), "/results/results_csv/")
filename <- paste0(file_path, "simulation_results_blocks_100_groups_n_100_p_50_1::1_non_stack_cpi")

N_CPU <- 100

list_func <- c(
    "Marg",
    "Permfit-DNN",
    "CPI-DNN",
    "CPI-RF",
    "gpfi",
    "gopfi",
    "dgi",
    "goi"
)

run_plot_auc <- TRUE
run_plot_type1error <- TRUE
run_plot_power <- TRUE
run_time <- TRUE

filename_lbl <- strsplit(filename, "_results_")[[1]][2]

if (run_plot_auc) {
      plot_method(paste0(filename, ".csv"),
          paste0("AUC_", filename_lbl),
          compute_auc,
          title = "AUC",
          list_func = list_func
      )
}


if (run_plot_type1error) {
      plot_method(paste0(filename, ".csv"),
          paste0("type1error_", filename_lbl),
          compute_pval,
          upper_bound = 0.05,
          title = "Type I Error",
          list_func = list_func
      )
}


if (run_plot_power) {
      plot_method(paste0(filename, ".csv"),
          paste0("power_", filename_lbl),
          compute_power,
          upper_bound = 0.05,
          title = "Power",
          list_func = list_func
      )
}


if (run_time) {
      plot_time(paste0(filename, ".csv"),
          paste0("time_bars_", filename_lbl),
          list_func = list_func,
          N_CPU = N_CPU
      )
}

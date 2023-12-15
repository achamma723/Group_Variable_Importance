DEBUG <- FALSE
N_SIMULATIONS <- `if`(!DEBUG, 1L:100L, 1L)
N_CPU <- ifelse(!DEBUG, 100L, 1L)

suppressMessages({
  require(data.table)
  if (!DEBUG) {
    require(snowfall)
    sfInit(parallel = TRUE, cpus = N_CPU, type = "SOCK")
    sfLibrary(gtools)
    sfLibrary(mlr)
    sfLibrary(snowfall)
    sfLibrary(reticulate)
    sfSource("data/data_gen.R")
    sfSource("utils/compute_methods.R")
  } else {
    library(gtools)
    library(mlr)
    library(reticulate)
    source("data/data_gen.R")
    source("utils/gimp.R")
    source("utils/compute_methods.R")
  }
})

my_apply <- lapply
if (!DEBUG) {
  my_apply <- sfLapply
}

##### Running Methods #####

methods <- c(
  "marginal",
  "permfit",
  "cpi",
  "cpi_rf",
  "gpfi",
  "gopfi",
  "dgi",
  "goi"
)

##### Configuration #####

param_grid <- expand.grid(
  # File, if given, for the real data
  file = "",
  # The file to regenerate samples with same covariance, if given
  sigma = "",
  # The number of samples
  n_samples = ifelse(!DEBUG, 1000L, 100L),
  # The number of covariates
  n_features = ifelse(!DEBUG, 50L, 50L),
  # Whether to use or not grouped variables
  group_bool = c(
    TRUE
  ),
  # Whether to use the stacking method
  group_stack = c(
    TRUE,
    FALSE
  ),
  # The mean for the simulation
  mean = c(0),
  # The correlation coefficient
  rho = c(
    # 0,
    # 0.2,
    # 0.5,
    0.8
  ),
  # The correlation between the groups if group-based simulations
  rho_group = c(
    0,
    0.2,
    0.5,
    0.8
  ),
  # Number of blocks
  n_blocks = ifelse(!DEBUG, 10L, 2L),
  # Type of simulation
  type_sim = c("blocks_group"),
  # Signal-to-Noise ratio
  snr = c(4),
  # The task (computation of the response vector)
  prob_sim_data = c(
    "regression_group_sim_1"
  ),
  # The running methods implemented
  method = methods,
  # Number of permutations/samples for the DNN algos
  n_perm = c(100L)
)

param_grid <- param_grid[
  ((!param_grid$group_stack) &
    (param_grid$method %in% c(
      "marginal",
      "cpi_rf",
      "gpfi",
      "gopfi",
      "dgi",
      "goi"
    )) |
    (!param_grid$method %in% c(
          "marginal",
          "cpi_rf",
          "gpfi",
          "gopfi",
          "dgi",
          "goi"
        ))
    ),
]

param_grid$index_i <- 1:nrow(param_grid)
cat(sprintf("Number of rows: %i \n", nrow(param_grid)))

if (!DEBUG) {
  sfExport("param_grid")
}

compute_method <- function(method,
                           index_i,
                           n_simulations, ...) {
  print("Begin")
  cat(sprintf("%s: %i \n", method, index_i))

  compute_fun <- function(seed, ...) {
    sim_data <- generate_data(
      seed,
      ...
    )
    print("Done loading data!")

    # Prepare the list of grouped labels
    if (list(...)$group_bool) {
      list_grps <- generate_grps(list(...)$p, list(...)$n_blocks)
    }

    timing <- system.time(
      out <- switch(as.character(method),
        marginal = compute_marginal(
          sim_data,
          list_grps = list_grps,
          ...
        ),
        permfit = compute_permfit(
          sim_data,
          seed,
          nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
          list_grps = list_grps,
          ...
        ),
        cpi = compute_cpi(
          sim_data,
          seed,
          nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
          list_grps = list_grps,
          ...
        ),
        cpi_rf = compute_cpi_rf(
          sim_data,
          seed,
          nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
          list_grps = list_grps,
          ...
        ),
        gpfi = compute_grps(
          sim_data,
          seed,
          list_grps = list_grps,
          func = "gpfi",
          ...
        ),
        gopfi = compute_grps(
          sim_data,
          seed,
          list_grps = list_grps,
          func = "gopfi",
          ...
        ),
        dgi = compute_grps(
          sim_data,
          seed,
          list_grps = list_grps,
          func = "dgi",
          ...
        ),
        goi = compute_grps(
          sim_data,
          seed,
          list_grps = list_grps,
          func = "goi",
          ...
        )
      )
    )
    out <- data.frame(out)
    out$elapsed <- timing[[3]]
    out$correlation <- list(...)$rho
    out$correlation_group <- list(...)$rho_group
    out$n_samples <- list(...)$n
    out$prob_data <- list(...)$prob_sim_data
    out$group_based <- list(...)$group_bool
    out$group_stack <- list(...)$group_stack
    return(out)
  }
  sim_range <- n_simulations
  # compute results
  result <- my_apply(sim_range, compute_fun, ...)
  # postprocess and package outputs
  result <- do.call(rbind, lapply(sim_range, function(ii) {
    out <- result[[ii - min(sim_range) + 1]]
    out$iteration <- ii
    out
  }))

  res <- data.table(result)[,
    mean(elapsed),
    by = .(
      n_samples,
      correlation,
      method,
      iteration,
      prob_data
    )
  ]

  res <- res[,
    sum(V1) / (N_CPU * 60),
    by = .(
      n_samples,
      method,
      correlation,
      prob_data
    )
  ]

  print(res)
  print("Finish")

  return(result)
}


if (DEBUG) {
  set.seed(42)
  param_grid <- param_grid[sample(1:nrow(param_grid), 5), ]
}

results <-
  by(
    param_grid, 1:nrow(param_grid),
    function(x) {
      with(
        x,
        compute_method(
          file = file,
          n = n_samples,
          p = n_features,
          group_bool = group_bool,
          group_stack = group_stack,
          mean = mean,
          rho = rho,
          rho_group = rho_group,
          sigma = sigma,
          n_blocks = n_blocks,
          type_sim = type_sim,
          snr = snr,
          method = method,
          index_i = index_i,
          n_simulations = N_SIMULATIONS,
          prob_sim_data = prob_sim_data,
          prob_type = strsplit(as.character(prob_sim_data), "_")[[1]][1],
          n_perm = n_perm
        )
      )
    }
  )

results <- rbindlist(results, fill=TRUE)

out_fname <- paste0(getwd(), "/results/results_csv/", "simulation_results_blocks_100_grps.csv")


if (DEBUG) {
  out_fname <- gsub("\\.csv", "-debug.csv", out_fname)
}

fwrite(results, out_fname)

if (!DEBUG) {
  sfStop()
}
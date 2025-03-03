# R wrapper for the RPAGP C++ implementation


# Set number of threads for parallel processing
setThreadOptions(numThreads = parallel::detectCores() - 1)



# wrapper function 
fit_rpagp <- function(y, n_iter, theta0, hyperparam, pinned_point, pinned_value = 1) {
  # Extract parameters
  initial_beta <- theta0$beta
  initial_tau <- theta0$tau
  initial_rho <- theta0$rho
  initial_sigma <- theta0$sigma
  initial_phi <- theta0$phi
  
  # Extract hyperparameters
  tau_prior_sd <- hyperparam$tau_prior_sd
  tau_proposal_sd <- hyperparam$tau_proposal_sd
  rho_prior_shape <- hyperparam$rho_prior_shape
  rho_prior_scale <- hyperparam$rho_prior_scale
  rho_proposal_sd <- hyperparam$rho_proposal_sd
  beta_prior_mu <- hyperparam$beta_prior_mu
  beta_prior_sd <- hyperparam$beta_prior_sd
  
  start <- Sys.time()
  
  # Call the C++ function
  results <- fit_rpagp_cpp(y, n_iter, 
                           initial_beta, initial_tau, 
                           initial_rho, initial_sigma, initial_phi,
                           tau_prior_sd, tau_proposal_sd,
                           rho_prior_shape, rho_prior_scale, rho_proposal_sd,
                           beta_prior_mu, beta_prior_sd,
                           pinned_point, pinned_value)
  
  
  end <- Sys.time()
  runtime <- end - start
  results$runtime <- runtime
  
  #a function to extract parameter values at a specific iteration
  extract_iteration <- function(iter) {
    list(
      beta = results$beta[, iter],
      tau = results$tau[, iter],
      rho = results$rho[iter],
      sigma = results$sigma[iter],
      phi = results$phi[, iter]
    )
  }

  chain <- vector("list", n_iter)
  chain_f <- vector("list", n_iter)
  
  for (i in 1:n_iter) {
    chain[[i]] <- extract_iteration(i)
    chain_f[[i]] <- results$f[, i]
  }
  
  return(list(
    chain = chain,
    chain_f = chain_f,
    beta = results$beta,
    tau = results$tau,
    rho = results$rho,
    sigma = results$sigma,
    phi = results$phi,
    f = results$f,
    runtime = runtime
  ))
}

# Function to generate synthetic data
generate_data <- function(n, n_time, theta) {
  results <- generate_data_cpp(n, n_time, theta$beta, theta$tau, 
                               theta$rho, theta$sigma, theta$phi)
  
  return(list(
    y = results$y,
    f = results$f,
    z = results$z,
    mu = results$mu
  ))
}

# Function to summarize MCMC output
getSummaryOutput <- function(results, dat_trials, y, burn_in)
{
  n <- dim(y)[2]
  n_time <- dim(y)[1]
  n_iter <- length(results$chain)
  n_final <- length((burn_in+1):n_iter)
  
  y_hat <- getSingleTrialEstimates(results, burn_in)
  probs = c(0.025, 0.5, 0.975)
  
  y_hat_quantiles <- array(NA, dim= c(3, n_time, n))
  for (ii in 1:n) {
    y_hat_quantiles[, , ii] = sapply(1:n_time, function(t) quantile(y_hat[t, ii, ], 
                                                                    probs = probs))
  }
  lower <- reshape2::melt(y_hat_quantiles[1, , ], varnames = c("time", "trial"))$value
  median <- reshape2::melt(y_hat_quantiles[2, , ], varnames = c("time", "trial"))$value
  upper <- reshape2::melt(y_hat_quantiles[3, , ], varnames = c("time", "trial"))$value
  
  out <- dat_trials %>% mutate(lwr = lower, med = median, upr = upper)
  
  return(tibble(out))
}


getSingleTrialEstimates <- function(results, burn_in)
{
  n_iter <- length(results$chain)
  n_final <- length((burn_in+1):n_iter)
  
  # - beta 
  chain_beta_burned = matrix(NA, n, n_final)
  ss <- 1
  for (tt in (burn_in+1):n_iter) {
    chain_beta_burned[, ss] <- results$chain[[tt]]$beta 
    ss <- ss + 1
  }
  # - f 
  chain_f_burned = matrix(NA, n_time, n_final)
  ss <- 1
  for (tt in (burn_in+1):n_iter) {
    chain_f_burned[, ss] <- results$chain_f[[tt]]
    ss <- ss + 1
  }
  # - 
  y_hat = array(NA, dim = c(n_time, n, n_final))
  for (tt in 1:n_final) {
    for (ii in 1:n) {
      y_hat[, ii, tt] = chain_beta_burned[ii, tt] * chain_f_burned[, tt]
    }
  }
  return(y_hat)
}


plot_rpagp <- function(results, true_f = NULL, observed_data = NULL, 
                       n_samples_to_show = 20, burnin = 0.3) {
  
  n_iter <- ncol(results$f)
  n_time <- nrow(results$f)
  time_points <- seq(0, 1, length.out = n_time)
  
  pinned_point <- floor(n_time/2)
  burnin_samples <- round(n_iter * burnin)
  post_burnin <- (burnin_samples + 1):n_iter
  
  #posterior mean
  f_samples <- results$f[, post_burnin]
  f_posterior_mean <- rowMeans(f_samples)
  
  # Sample a subset of iterations to plot
  if(length(post_burnin) > n_samples_to_show) {
    selected_samples <- sample(post_burnin, n_samples_to_show)
  } else {
    selected_samples <- post_burnin
  }
  
  #data frame for samples
  samples_df <- data.frame(
    time = rep(time_points, length(selected_samples)),
    iteration = rep(selected_samples, each = n_time),
    f_value = as.vector(results$f[, selected_samples])
  )
  
  p <- ggplot() +
    geom_line(data = samples_df, 
              aes(x = time, y = f_value, group = iteration),
              color = "gray70", alpha = 0.6) +
    theme_minimal() +
    theme(
      panel.border = element_rect(color = "black", fill = NA),
      legend.position = "none"  
    ) +
    labs(x = "Time", y = "f(t)")
  
  # Add true f 
  if (!is.null(true_f)) {
    # Normalize to match pinned_value = 1 at pinned_point
    true_f <- true_f / true_f[pinned_point]
    
    p <- p + 
      geom_line(
        data = data.frame(time = time_points, f_value = true_f),
        aes(x = time, y = f_value), 
        color = "limegreen", 
        size = 1.2
      )
  }
  
  # Add posterior mean
  p <- p + 
    geom_line(
      data = data.frame(time = time_points, f_value = f_posterior_mean),
      aes(x = time, y = f_value), 
      color = "darkgreen", 
      linetype = "dashed",
      size = 1
    )
  
  if (!is.null(observed_data) && is.matrix(observed_data)) {
    emp_mean <- rowMeans(observed_data)
    
    p <- p + 
      geom_point(
        data = data.frame(time = time_points, f_value = emp_mean),
        aes(x = time, y = f_value),
        color = "black",
        size = 1.2,
        shape = 16
      )
  }
  
  return(p)
}


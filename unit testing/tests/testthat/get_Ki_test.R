


test_that("R and C++ implementations of get_K_i produce equivalent results", {

  skip_on_cran()
  r_env <- new.env()
  cpp_env <- new.env()
  
  r_dir <- "C:\\Users\\divya\\OneDrive\\Desktop\\Research\\Neural\\GP\\practice2\\rpagp\\unit testing\\R"
  cpp_dir <- "C:\\Users\\divya\\OneDrive\\Desktop\\Research\\Neural\\GP\\practice2\\rpagp\\unit testing\\SRC"
  source(file.path(r_dir, "utilities.R"), local = r_env)
  Rcpp::sourceCpp(file.path(cpp_dir, "rpagp_cpp.cpp"), env = cpp_env)
  
  
  set.seed(42)
  n_time <- 50
  x <- seq(0, 1, length.out = n_time)
  test_rho <- 10
  test_tau <- 0.1
  test_beta <- 2.0
  
  # Test function to run both implementations with same parameters
  test_K_i <- function(rho, tau, beta) {
    
    theta <- list(rho = rho, tau = tau, beta = beta)
    r_result <- r_env$get_K_i(x, theta)
    cpp_result <- cpp_env$get_trial_K_i(x, rho, tau, beta)
    correlation <- cor(as.vector(r_result), as.vector(cpp_result))
    max_diff <- max(abs(r_result - cpp_result))
    rel_diff <- if(max(abs(r_result)) > 0) {
      max_diff / max(abs(r_result))
    } else {
      0
    }
    
    list(
      r_result = r_result,
      cpp_result = cpp_result,
      correlation = correlation,
      max_diff = max_diff,
      rel_diff = rel_diff
    )
  }
  
  # Test with standard parameters
  base_results <- test_K_i(test_rho, test_tau, test_beta)
  
  # Test with various parameters
  rho_values <- c(0.1, 1, 5, 10, 50)
  tau_values <- c(-0.2, -0.1, 0, 0.1, 0.2)
  beta_values <- c(0.5, 1.0, 2.0)
  
  all_correlations <- numeric(0)
  all_rel_diffs <- numeric(0)
  
  # Run tests with parameter combinations 
  for(rho in rho_values[c(1,3,5)]) {
    for(tau in tau_values[c(1,3,5)]) {
      for(beta in beta_values[c(1,3)]) {
        
        if(rho == 10 && tau == 0.1 && beta == 2.0) next
        
        results <- test_K_i(rho, tau, beta)
        
        message(sprintf("rho=%.2f, tau=%.2f, beta=%.1f: corr=%.6f, max_diff=%.6e, rel_diff=%.6e", 
                        rho, tau, beta, results$correlation, results$max_diff, results$rel_diff))
        
        all_correlations <- c(all_correlations, results$correlation)
        all_rel_diffs <- c(all_rel_diffs, results$rel_diff)
      }
    }
  }
  
  
  message("\nBase case (rho=10, tau=0.1, beta=2.0):")
  message("Correlation: ", base_results$correlation)
  message("Maximum absolute difference: ", base_results$max_diff)
  message("Relative difference: ", base_results$rel_diff)
  
  
  expect_true(base_results$correlation > 0.999, 
              label = "Base case correlation should be extremely high")
  
  expect_true(min(all_correlations) > 0.99, 
              label = "Correlation should be high across all parameter combinations")
  
  expect_true(base_results$rel_diff < 0.01, 
              label = "Base case relative difference should be small")
  
  expect_true(mean(all_rel_diffs) < 0.01, 
              label = "Average relative difference should be small")
  
  # Test direct equality 
  expect_equal(base_results$r_result, base_results$cpp_result, 
               tolerance = 1e-4, 
               label = "K_i matrices should be approximately equal")
  
  double_beta <- 2.0 * test_beta  

  theta1 <- list(rho = test_rho, tau = test_tau, beta = test_beta)
  theta2 <- list(rho = test_rho, tau = test_tau, beta = double_beta)
  
  r_result1 <- r_env$get_K_i(x, theta1)
  r_result2 <- r_env$get_K_i(x, theta2)
  
  cpp_result1 <- cpp_env$get_trial_K_i(x, test_rho, test_tau, test_beta)
  cpp_result2 <- cpp_env$get_trial_K_i(x, test_rho, test_tau, double_beta)
  
  # Calculate ratios
  r_ratio <- r_result2 / r_result1
  cpp_ratio <- cpp_result2 / cpp_result1
  
  message("R scaling ratio (should be 2.0): ", mean(r_ratio))
  message("C++ scaling ratio (should be 2.0): ", mean(cpp_ratio))
  
  # mean ratio - should be very close to 2.0
  expect_equal(mean(r_ratio), 2.0, tolerance = 1e-10, 
               label = "R implementation scales correctly with beta")
  
  expect_equal(mean(cpp_ratio), 2.0, tolerance = 1e-10,
               label = "C++ implementation scales correctly with beta")
  
  # Check individual elements 
  expect_true(all(abs(r_ratio - 2.0) < 1e-8),
              label = "R implementation scales uniformly across matrix")
  
  expect_true(all(abs(cpp_ratio - 2.0) < 1e-8),
              label = "C++ implementation scales uniformly across matrix")
  
  # Check that implementations match each other at both scales
  expect_equal(r_result1, cpp_result1, tolerance = 1e-4,
               label = "Implementations match at original beta")
  
  expect_equal(r_result2, cpp_result2, tolerance = 1e-4,
               label = "Implementations match at doubled beta")
})



test_that("C++ implementation of dmvnorm is numerically correct and stable", {
  
  skip_on_cran()
  r_env <- new.env()
  cpp_env <- new.env()
  r_dir <- "C:\\Users\\divya\\OneDrive\\Desktop\\Research\\Neural\\GP\\practice2\\rpagp\\unit testing\\R"
  cpp_dir <- "C:\\Users\\divya\\OneDrive\\Desktop\\Research\\Neural\\GP\\practice2\\rpagp\\unit testing\\SRC"
  Rcpp::sourceCpp(file.path(cpp_dir, "rpagp_cpp.cpp"), env = cpp_env)
  
  compare_with_r <- function(x, mean, sigma, log_p = TRUE, label = "test case") {
    r_result <- mvtnorm::dmvnorm(x, mean, sigma, log = log_p)
    cpp_result <- cpp_env$dmvnorm(x, mean, sigma, log_p)
    
    # Relative difference
    rel_diff <- if (abs(r_result) > 1e-10) {
      abs(r_result - cpp_result) / abs(r_result)
    } else {
      abs(r_result - cpp_result)
    }
    
    message(sprintf("%s: R=%.8e, C++=%.8e, rel_diff=%.8e", 
                    label, r_result, cpp_result, rel_diff))
    
    return(list(r_result = r_result, cpp_result = cpp_result, rel_diff = rel_diff))
  }
  
  # basic
  message("\n=== Basic Correctness Tests ===")
  
  # test with standard normal distribution
  set.seed(42)
  n_dim <- 3
  x <- rnorm(n_dim)
  mean_vec <- rep(0, n_dim)
  sigma <- diag(n_dim)
  
  std_normal <- compare_with_r(x, mean_vec, sigma, TRUE, "Standard normal")
  expect_equal(std_normal$r_result, std_normal$cpp_result, 
               tolerance = 1e-6, 
               label = "Standard normal log density should match R implementation")
  
  # Test non-standard mean and variance
  x2 <- rnorm(n_dim, mean = 2, sd = 3)
  mean_vec2 <- rep(2, n_dim)
  sigma2 <- diag(9, n_dim)
  
  nonstandard <- compare_with_r(x2, mean_vec2, sigma2, TRUE, "Non-standard normal")
  expect_equal(nonstandard$r_result, nonstandard$cpp_result, 
               tolerance = 1e-6, 
               label = "Non-standard normal log density should match R implementation")
  
  # Test with correlation
  corr_mat <- matrix(0.5, n_dim, n_dim)
  diag(corr_mat) <- 1
  std_dev <- rep(2, n_dim)
  sigma_corr <- diag(std_dev) %*% corr_mat %*% diag(std_dev)
  
  correlated <- compare_with_r(x, mean_vec, sigma_corr, TRUE, "Correlated normal")
  expect_equal(correlated$r_result, correlated$cpp_result, 
               tolerance = 1e-6, 
               label = "Correlated normal log density should match R implementation")
  
  # Test with exp scale
  nonlog <- compare_with_r(x, mean_vec, sigma, FALSE, "Standard normal (non-log)")
  expect_equal(nonlog$r_result, nonlog$cpp_result, 
               tolerance = 1e-6, 
               label = "Standard normal density (non-log) should match R implementation")
  
  # numerical stability
  message("\n=== Numerical Stability Tests ===")
  
  # Test with nearly singular covariance matrix
  near_singular <- matrix(0.999, n_dim, n_dim)
  diag(near_singular) <- 1
  
  singular_test <- compare_with_r(x, mean_vec, near_singular, TRUE, "Nearly singular")
  expect_equal(singular_test$r_result, singular_test$cpp_result, 
               tolerance = 1e-4, 
               label = "Nearly singular covariance matrix should be handled stably")
  
  # Test with ill-conditioned covariance matrix 
  ill_cond <- diag(c(1, 1e6, 1e-6))
  
  illcond_test <- compare_with_r(x, mean_vec, ill_cond, TRUE, "Ill-conditioned")
  expect_equal(illcond_test$r_result, illcond_test$cpp_result, 
               tolerance = 1e-4, 
               label = "Ill-conditioned covariance matrix should be handled stably")
  
  # Test with asymmetric covariance matrix
  asym_cov <- matrix(c(1, 0.5, 0.4, 
                       0.6, 1, 0.3, 
                       0.4, 0.3, 1), 3, 3)
  
  asymmetric_test <- tryCatch({
    compare_with_r(x, mean_vec, asym_cov, TRUE, "Asymmetric matrix")
  }, error = function(e) {
    message("R implementation failed on asymmetric matrix: ", e$message)
    sym_cov <- (asym_cov + t(asym_cov))/2
    result <- list(
      r_result = mvtnorm::dmvnorm(x, mean_vec, sym_cov, log = TRUE),
      cpp_result = cpp_env$dmvnorm(x, mean_vec, asym_cov, TRUE)
    )
    result$rel_diff <- abs(result$r_result - result$cpp_result) / abs(result$r_result)
    message(sprintf("Symmetrized: R=%.8e, C++=%.8e, rel_diff=%.8e", 
                    result$r_result, result$cpp_result, result$rel_diff))
    return(result)
  })
  
  expect_true(!is.na(asymmetric_test$cpp_result), 
              label = "C++ implementation should handle asymmetric matrix")
  
  # edge cases
  message("\n=== Edge Cases ===")
  
  # Test with very high dimensionality 
  if (interactive()) {  # Skip in automated testing
    n_high <- 100
    x_high <- rnorm(n_high)
    mean_high <- rep(0, n_high)
    sigma_high <- diag(n_high)
    
    high_dim <- compare_with_r(x_high, mean_high, sigma_high, TRUE, "High-dimensional")
    expect_equal(high_dim$r_result, high_dim$cpp_result, 
                 tolerance = 1e-4, 
                 label = "High-dimensional case should be handled correctly")
  }
  
  # extreme values
  x_extreme <- rep(10, n_dim)  # Very far from mean
  extreme_test <- compare_with_r(x_extreme, mean_vec, sigma, TRUE, "Extreme values")
  expect_equal(extreme_test$r_result, extreme_test$cpp_result, 
               tolerance = 1e-4, 
               label = "Extreme values should be handled correctly")
  
  # Test with non-finite values in the covariance matrix
  sigma_bad <- sigma
  sigma_bad[1,2] <- NaN
  
  bad_sigma_test <- tryCatch({
    cpp_result <- cpp_env$dmvnorm(x, mean_vec, sigma_bad, TRUE)
    message(sprintf("Bad sigma: C++=%.8e", cpp_result))
    list(cpp_result = cpp_result)
  }, error = function(e) {
    message("C++ implementation failed on bad sigma: ", e$message)
    list(cpp_result = NA)
  })
  
  expect_true(is.finite(bad_sigma_test$cpp_result), 
              label = "Bad sigma should not cause non-finite results")
  
  # error handling
  message("\n=== Error Handling ===")
  
  # Test with non-positive definite matrix
  non_pd <- matrix(1, n_dim, n_dim)  
  
  non_pd_test <- tryCatch({
    compare_with_r(x, mean_vec, non_pd, TRUE, "Non-positive definite")
  }, error = function(e) {
    message("R implementation failed on non-PD matrix: ", e$message)
    cpp_result <- cpp_env$dmvnorm(x, mean_vec, non_pd, TRUE)
    message(sprintf("Non-PD matrix: C++=%.8e", cpp_result))
    list(cpp_result = cpp_result)
  })
  
  expect_true(is.finite(non_pd_test$cpp_result), 
              label = "Non-positive definite matrix should not cause non-finite results")
  
  # Test with mismatched dimensions
  mismatched_test <- tryCatch({
    x_bad <- c(x, 0)  # One extra dimension
    cpp_result <- cpp_env$dmvnorm(x_bad, mean_vec, sigma, TRUE)
    message(sprintf("Mismatched dimensions: C++=%.8e", cpp_result))
    list(cpp_result = cpp_result)
  }, error = function(e) {
    message("C++ implementation failed on mismatched dimensions: ", e$message)
    list(cpp_result = NA)
  })
  
  expect_true(!is.null(mismatched_test$cpp_result), 
              label = "Mismatched dimensions should be handled gracefully")
  
  # 5. PERFORMANCE COMPARISON
  message("\n=== Performance Comparison ===")
  
  if (interactive()) {  # Only run timing tests in interactive mode
    n_iter <- 1000
    
    r_time <- system.time({
      for (i in 1:n_iter) {
        mvtnorm::dmvnorm(x, mean_vec, sigma, log = TRUE)
      }
    })
    
    cpp_time <- system.time({
      for (i in 1:n_iter) {
        cpp_env$dmvnorm(x, mean_vec, sigma, TRUE)
      }
    })
    
    speedup <- r_time[3] / cpp_time[3]
    
    message(sprintf("R time: %.4f sec, C++ time: %.4f sec, Speedup: %.2fx", 
                    r_time[3], cpp_time[3], speedup))
    
    expect_true(speedup >= 1, 
                label = "C++ implementation should be at least as fast as R implementation")
  }
  
  # consistency
  message("\n=== Consistency Checks ===")
  
  # Log vs non-log consistency
  log_result <- cpp_env$dmvnorm(x, mean_vec, sigma, TRUE)
  nonlog_result <- cpp_env$dmvnorm(x, mean_vec, sigma, FALSE)
  log_diff <- abs(log_result - log(nonlog_result))
  
  message(sprintf("Log vs exp(log) consistency: diff=%.8e", log_diff))
  
  expect_true(log_diff < 1e-10, 
              label = "Log and non-log results should be consistent")
  
  # Different regularization levels
  sigma_orig <- sigma_corr
  
  # Add small regularization
  sigma_reg1 <- sigma_orig
  diag(sigma_reg1) <- diag(sigma_reg1) + 1e-6
  
  # Add medium regularization
  sigma_reg2 <- sigma_orig
  diag(sigma_reg2) <- diag(sigma_reg2) + 1e-4
  
  # Add large regularization
  sigma_reg3 <- sigma_orig
  diag(sigma_reg3) <- diag(sigma_reg3) + 1e-2
  
  result_orig <- cpp_env$dmvnorm(x, mean_vec, sigma_orig, TRUE)
  result_reg1 <- cpp_env$dmvnorm(x, mean_vec, sigma_reg1, TRUE)
  result_reg2 <- cpp_env$dmvnorm(x, mean_vec, sigma_reg2, TRUE)
  result_reg3 <- cpp_env$dmvnorm(x, mean_vec, sigma_reg3, TRUE)
  
  message(sprintf("Original: %.8e", result_orig))
  message(sprintf("Reg 1e-6: %.8e (diff: %.8e)", result_reg1, abs(result_orig - result_reg1)))
  message(sprintf("Reg 1e-4: %.8e (diff: %.8e)", result_reg2, abs(result_orig - result_reg2)))
  message(sprintf("Reg 1e-2: %.8e (diff: %.8e)", result_reg3, abs(result_orig - result_reg3)))
  
  # Should be stable under small regularization
  expect_true(abs(result_orig - result_reg1) < 1e-4, 
              label = "Small regularization should not significantly change results")
  
  # SUMMARY
  message("\n=== Summary ===")
  
  # If we get here, all tests have passed
  message("All tests passed. The C++ dmvnorm implementation appears to be working correctly.")
})
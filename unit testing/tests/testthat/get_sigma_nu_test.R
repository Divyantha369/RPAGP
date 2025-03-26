

ultosymmetric <- function(m) {
  m = m + t(m) - diag(diag(m))
  return(m)
}


test_that("Matrix properties: AR(1) process", {
  skip_on_cran()
  
  
  cpp_env <<- new.env()
  cpp_dir <- "C:\\Users\\divya\\OneDrive\\Desktop\\Research\\Neural\\GP\\practice2\\rpagp\\unit testing\\SRC"
  Rcpp::sourceCpp(file.path(cpp_dir, "rpagp_cpp.cpp"), env = cpp_env)
  
  
  # standard AR(1)
  n_time <- 10
  phi <- c(0.7)
  sigma <- 0.5
  
  
  cpp_sigma_nu <- cpp_env$get_Sigma_nu(phi, sigma, n_time)
  expect_equal(dim(cpp_sigma_nu), c(n_time, n_time))
  expect_equal(cpp_sigma_nu, t(cpp_sigma_nu), tolerance = 1e-10)
  
  # Test Toeplitz structure 
  for (k in 1:(n_time-1)) {
    diag_values <- diag(cpp_sigma_nu[-(1:k), 1:(n_time-k)])
    expect_true(all(abs(diag_values - diag_values[1]) < 1e-10))
  }
  
  
  eigenvalues <- eigen(cpp_sigma_nu, only.values = TRUE)$values
  expect_true(all(eigenvalues > 0))
  cond_number <- max(eigenvalues) / min(eigenvalues)
  expect_lt(cond_number, 1000)  
  
  # Test first row values follow exponential decay pattern for AR(1)
  first_row <- cpp_sigma_nu[1, 1:5]
  ratios <- first_row[2:5] / first_row[1:4]
  ratio_diff <- abs(ratios - phi[1])
  expect_true(all(ratio_diff < 0.05))
})

test_that("Matrix properties: AR(2) process", {
  skip_on_cran()
  
  # standard AR(2)
  n_time <- 20
  phi <- c(0.5, 0.2)
  sigma <- 0.3
  

  cpp_sigma_nu <- cpp_env$get_Sigma_nu(phi, sigma, n_time)
  expect_equal(dim(cpp_sigma_nu), c(n_time, n_time))
  expect_equal(cpp_sigma_nu, t(cpp_sigma_nu), tolerance = 1e-10)
  
  # Test Toeplitz structure
  for (k in 1:5) {  
    diag_values <- diag(cpp_sigma_nu[-(1:k), 1:(n_time-k)])
    expect_true(all(abs(diag_values - diag_values[1]) < 1e-10))
  }
  
  # Test positive-definiteness
  eigenvalues <- eigen(cpp_sigma_nu, only.values = TRUE)$values
  expect_true(all(eigenvalues > 0))
  
  cond_number <- max(eigenvalues) / min(eigenvalues)
  expect_lt(cond_number, 1000)  
  
  # Test that autocovariance follows AR(2) difference equation
  
  for (k in 3:10) {
    predicted <- phi[1] * cpp_sigma_nu[1, k-1] + phi[2] * cpp_sigma_nu[1, k-2]
    actual <- cpp_sigma_nu[1, k]
    expect_equal(predicted, actual, tolerance = 0.05)
  }
})

test_that("Matrix properties: Higher-order AR process", {
  skip_on_cran()
  
  # parameters - AR(3)
  n_time <- 25
  phi <- c(0.3, -0.2, 0.1)
  sigma <- 0.4
  
  
  cpp_sigma_nu <- cpp_env$get_Sigma_nu(phi, sigma, n_time)
  expect_equal(dim(cpp_sigma_nu), c(n_time, n_time))
  expect_equal(cpp_sigma_nu, t(cpp_sigma_nu), tolerance = 1e-10)
  
  # Test Toeplitz structure 
  for (k in 1:5) {  
    diag_values <- diag(cpp_sigma_nu[-(1:k), 1:(n_time-k)])
    expect_true(all(abs(diag_values - diag_values[1]) < 1e-10))
  }
  
  
  eigenvalues <- eigen(cpp_sigma_nu, only.values = TRUE)$values
  expect_true(all(eigenvalues > 0))
  
  # Test that autocovariance follows AR(3) difference equation
 
  for (k in 4:10) {
    predicted <- phi[1] * cpp_sigma_nu[1, k-1] + 
      phi[2] * cpp_sigma_nu[1, k-2] + 
      phi[3] * cpp_sigma_nu[1, k-3]
    actual <- cpp_sigma_nu[1, k]
    expect_equal(predicted, actual, tolerance = 0.05)
  }
})

test_that("Edge cases: Near non-stationary AR(1) process", {
  skip_on_cran()
  
  # very close to non-stationarity
  n_time <- 10
  phi <- c(0.995)
  sigma <- 0.5
  
  cpp_sigma_nu <- cpp_env$get_Sigma_nu(phi, sigma, n_time)
  expect_equal(dim(cpp_sigma_nu), c(n_time, n_time))
  
  expect_equal(cpp_sigma_nu, t(cpp_sigma_nu), tolerance = 1e-10)
  
  eigenvalues <- eigen(cpp_sigma_nu, only.values = TRUE)$values
  expect_true(all(eigenvalues > 0))
  
  # Test that values are bounded 
  expect_true(all(cpp_sigma_nu < 1000 * sigma^2))
  
  # Test that diagonal elements are positive
  diag_elements <- diag(cpp_sigma_nu)
  expect_true(all(diag_elements > 0))
})

test_that("Edge cases: Near non-stationary AR(2) process", {
  skip_on_cran()
  
  # Test parameters - close to the stationarity boundary
  n_time <- 10
  phi <- c(0.7, 0.28)  # Close to φ₁ + φ₂ = 1
  sigma <- 0.5
  
  # Generate matrix with C++ implementation
  cpp_sigma_nu <- cpp_env$get_Sigma_nu(phi, sigma, n_time)
  
  # Test dimension
  expect_equal(dim(cpp_sigma_nu), c(n_time, n_time))
  
  # Test symmetry
  expect_equal(cpp_sigma_nu, t(cpp_sigma_nu), tolerance = 1e-10)
  
  # Test positive-definiteness
  eigenvalues <- eigen(cpp_sigma_nu, only.values = TRUE)$values
  expect_true(all(eigenvalues > 0))
  
  # Test that values are bounded (shouldn't explode to infinity)
  expect_true(all(cpp_sigma_nu < 1000 * sigma^2))
})

test_that("Consistency with theoretical values for AR(1)", {
  skip_on_cran()
  
  # AR(1) process with phi = 0.5
  n_time <- 10
  phi <- c(0.5)
  sigma <- 1.0
  
  # Generate matrix with C++ implementation
  cpp_sigma_nu <- cpp_env$get_Sigma_nu(phi, sigma, n_time)
  
  # For AR(1), we know the theoretical autocovariance values
  # γ(k) = σ² * φᵏ / (1-φ²)
  gamma_0 <- sigma^2 / (1 - phi[1]^2)
  theoretical_acf <- numeric(n_time)
  theoretical_acf[1] <- gamma_0
  for (k in 2:n_time) {
    theoretical_acf[k] <- gamma_0 * phi[1]^(k-1)
  }
  
  # Create theoretical covariance matrix
  theoretical_sigma <- toeplitz(theoretical_acf)
  
  # Test that matrices are close
  max_rel_diff <- max(abs(cpp_sigma_nu - theoretical_sigma) / theoretical_sigma)
  expect_lt(max_rel_diff, 0.01)  # Should be within 1%
})

test_that("Numerical stability with various sizes", {
  skip_on_cran()
  
  # Test with different matrix sizes
  test_sizes <- c(5, 20, 50, 100)
  
  for (n_time in test_sizes) {
    # AR(1) process
    phi <- c(0.6)
    sigma <- 0.4
    
    # C++ implementation
    expect_no_error(cpp_sigma_nu <- cpp_env$get_Sigma_nu(phi, sigma, n_time))
    
    # Check dimensions
    expect_equal(dim(cpp_sigma_nu), c(n_time, n_time))
    
    # Check positive definiteness
    # For large matrices, check only a few eigenvalues to save time
    if (n_time <= 50) {
      eigenvalues <- eigen(cpp_sigma_nu, only.values = TRUE)$values
      expect_true(all(eigenvalues > 0))
    } else {
      # For large matrices, check min eigenvalue using power method approximation
      min_eig <- min(svd(cpp_sigma_nu, nu = 0, nv = 0)$d)
      expect_true(min_eig > 0)
    }
  }
})

test_that("get_Sigma_nu preserves stationarity constraints", {
  skip_on_cran()
  
  # Test with extreme parameter values
  test_cases <- list(
    list(phi = c(1.2), sigma = 0.5, n_time = 10),              # Non-stationary AR(1)
    list(phi = c(0.9, 0.5), sigma = 0.4, n_time = 10),         # Non-stationary AR(2)
    list(phi = c(0.5, 0.6), sigma = 0.4, n_time = 10),         # Non-stationary AR(2)
    list(phi = c(0.3, -1.2), sigma = 0.4, n_time = 10),        # Non-stationary AR(2)
    list(phi = c(0.3, -0.2, 0.95), sigma = 0.4, n_time = 10)   # Extreme AR(3)
  )
  
  for (case in test_cases) {
    # Generate matrix - should not error despite non-stationary input
    expect_no_error(cpp_sigma_nu <- cpp_env$get_Sigma_nu(case$phi, case$sigma, case$n_time))
    
    # Check positive definiteness - matrix should be fixed to be valid
    eigenvalues <- eigen(cpp_sigma_nu, only.values = TRUE)$values
    expect_true(all(eigenvalues > 0))
  }
})
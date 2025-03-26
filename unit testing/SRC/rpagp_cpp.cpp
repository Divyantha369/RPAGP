// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

using namespace Rcpp;
using namespace arma;

// Constants for optimization
constexpr int PARALLEL_THRESHOLD = 25;  // Minimum size for parallelization
constexpr double NUMERICAL_STABILITY_NUGGET = 1e-6;  // Default matrix regularization

// Set the number of threads and configure OpenMP environment
void configureParallelEnvironment() {
  int max_threads = omp_get_max_threads();
  int use_threads = std::max(1, (int)(max_threads * 0.75));
  omp_set_num_threads(use_threads);
  
  // Disable nested parallelism by default
  omp_set_nested(0);
  
  // Set dynamic thread adjustment
  omp_set_dynamic(1);
  
  Rcout << "Using " << use_threads << " threads out of " << max_threads << " available\n";
}

// Thread-safe error storage
class ThreadSafeErrorCollector {
private:
  std::vector<std::string> errors;
  omp_lock_t lock;
  
public:
  ThreadSafeErrorCollector() {
    omp_init_lock(&lock);
  }
  
  ~ThreadSafeErrorCollector() {
    omp_destroy_lock(&lock);
  }
  
  void addError(const std::string& error) {
    omp_set_lock(&lock);
    errors.push_back(error);
    omp_unset_lock(&lock);
  }
  
  void reportErrors() {
    for (const auto& error : errors) {
      Rcout << error << "\n";
    }
    errors.clear();
  }
};

// Global error collector
ThreadSafeErrorCollector errorCollector;

// Check if matrix is positive definite
bool is_positive_definite(const arma::mat& M) {
  try {
    // Try Cholesky decomposition - will throw exception if not positive definite
    arma::mat L = arma::chol(M, "lower");
    return true;
  } catch (...) {
    return false;
  }
}

// Regularize matrix to ensure positive definiteness
arma::mat ensure_positive_definite(const arma::mat& M, double min_nugget = NUMERICAL_STABILITY_NUGGET) {
  if (is_positive_definite(M)) {
    return M;
  }
  
  arma::mat M_reg = M;
  
  // Find smallest eigenvalue
  arma::vec eigval;
  try {
    eigval = arma::eig_sym(M);
    double min_eig_val = arma::min(eigval);
    
    if (min_eig_val <= 0) {
      double eps = std::abs(min_eig_val) + min_nugget;
      M_reg.diag() += eps;
    }
  } catch (...) {
    // If eigenvalue computation fails, add a stronger regularization
    M_reg.diag() += 1e-4;
  }
  
  return M_reg;
}

// Robust matrix inversion with multiple fallback methods
arma::mat robust_inv(const arma::mat& A, double nugget = NUMERICAL_STABILITY_NUGGET) {
  // Strategy 1: Direct symmetric positive definite inversion
  try {
    return arma::inv_sympd(A);
  } catch (...) {
    // Continue to next method
  }
  
  // Strategy 2: Add regularization and try again
  try {
    arma::mat A_reg = A;
    A_reg.diag() += nugget;
    return arma::inv_sympd(A_reg);
  } catch (...) {
    // Continue to next method
  }
  
  // Strategy 3: Use eigendecomposition-based inversion
  try {
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, A);
    
    // Ensure positive eigenvalues with stronger threshold
    for (unsigned int j = 0; j < eigval.n_elem; j++) {
      eigval(j) = std::max(eigval(j), 1e-4);
    }
    
    // Reconstruct inverse from eigendecomposition
    return eigvec * arma::diagmat(1.0 / eigval) * eigvec.t();
  } catch (...) {
    // Continue to next method
  }
  
  // Strategy 4: Last resort - use pinv
  return arma::pinv(A);
}

// Square exponential kernel function with SIMD and cache optimization
// @param x Vector of time points
// @param rho Length scale parameter
// @param nugget Small value for numerical stability
// [[Rcpp::export]]
arma::mat sq_exp_kernel(const arma::vec& x, double rho, double alpha = 1.0, double nugget = NUMERICAL_STABILITY_NUGGET) {
  int n = x.n_elem;
  
  // Ensure rho is not too small
  double rho_safe = std::max(rho, 0.01);
  double rho2 = std::pow(rho_safe, 2);
  double alpha2 = std::pow(alpha, 2);
  
  // Create the vector of covariances for Toeplitz construction
  arma::vec v(n);
  
  // First calculate the vector for Toeplitz construction
  #pragma omp parallel for simd schedule(static)
  for (int i = 0; i < n; i++) {
    v(i) = alpha2 * std::exp(-rho2 / 2.0 * std::pow(x(i), 2));
  }
  
  // Create the Toeplitz matrix
  arma::mat K(n, n);
  
  if (n > PARALLEL_THRESHOLD) {
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        // Toeplitz pattern: elements depend on absolute difference of indices
        K(i, j) = v(std::abs(i - j));
      }
      // Add nugget to diagonal for numerical stability
      K(i, i) += nugget;
    }
  } else {
    // Sequential version for small matrices
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        K(i, j) = v(std::abs(i - j));
      }
      K(i, i) += nugget;
    }
  }
  
  return K;
}

// Get Covariance AR process with improved numerical stability
// @param phi autoregressive parms (p x 1)
// @param sigma error variance
// @param n_time number of time points
// [[Rcpp::export]]
arma::mat get_Sigma_nu(const arma::vec& phi, double sigma, int n_time) {
  int p = phi.n_elem;  // AR order
  
  // Restrict phi to stationary range
  arma::vec phi_safe = phi;
  if (p == 1) {
    phi_safe(0) = std::min(std::max(phi_safe(0), -0.999), 0.999);
  } 
  else if (p == 2) {
    // Apply constraints for AR(2) stationarity
    if (std::abs(phi_safe(0) + phi_safe(1)) > 0.999 || 
        std::abs(phi_safe(0) - phi_safe(1)) > 0.999 || 
        std::abs(phi_safe(1)) > 0.999) {
      phi_safe(0) *= 0.99;
      phi_safe(1) *= 0.99;
    }
  }
  
  // Compute autocorrelation function
  arma::vec acf = arma::zeros(n_time);
  double sigma2 = std::pow(sigma, 2);
  
  // Compute acf(0) based on AR order
  if (p == 1) {
    // AR(1) process
    acf(0) = sigma2 / (1.0 - std::pow(phi_safe(0), 2));
  } 
  else if (p == 2) {
    // AR(2) process
    double phi1 = phi_safe(0);
    double phi2 = phi_safe(1);
    double denom = (1.0 - phi2) * ((1.0 + phi2) * (1.0 - phi2) - std::pow(phi1, 2));
    denom = std::max(denom, 1e-6);  // Avoid division by zero
    acf(0) = sigma2 / denom;
    
    // Compute acf(1) directly for AR(2)
    acf(1) = phi1/(1.0 - phi2) * acf(0);
  }
  else {
    // General AR(p) process - use Yule-Walker equations
    double denom = 1.0;
    for (int i = 0; i < p; i++) {
      denom -= std::pow(phi_safe(i), 2);
    }
    denom = std::max(denom, 1e-6);  // Avoid division by zero
    acf(0) = sigma2 / denom;
  }
  
  // Recursively compute autocovariances at lags > 0 or > 1 (for AR(2))
  int start_lag = (p == 2) ? 2 : 1;
  for (int lag = start_lag; lag < n_time; lag++) {
    double sum = 0.0;
    for (int j = 0; j < p; j++) {
      if (lag - (j + 1) >= 0) {
        sum += phi_safe(j) * acf(lag - (j + 1));
      }
    }
    acf(lag) = sum;
  }
  
  // Build Toeplitz matrix efficiently
  arma::mat Sigma_nu(n_time, n_time);
  
  if (n_time > PARALLEL_THRESHOLD) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_time; i++) {
      for (int j = 0; j < n_time; j++) {
        int lag = std::abs(i - j);
        Sigma_nu(i, j) = acf(lag);
      }
    }
  } else {
    for (int i = 0; i < n_time; i++) {
      for (int j = 0; j < n_time; j++) {
        int lag = std::abs(i - j);
        Sigma_nu(i, j) = acf(lag);
      }
    }
  }
  
  // Ensure positive definiteness
  Sigma_nu = ensure_positive_definite(Sigma_nu);
  
  return Sigma_nu;
}

// Get Cov(y_i, f) - K_i matrix for a specific trial with vectorization
// [[Rcpp::export]]
arma::mat get_trial_K_i(const arma::vec& x, double rho, double tau, double beta) {
  int n_time = x.n_elem;
  arma::mat K(n_time, n_time);
  
  // Ensure rho is not too small
  double rho_safe = std::max(rho, 0.01);
  double rho2 = std::pow(rho_safe, 2);
  
  // Precompute shifted x values for better cache locality
  arma::vec x_shifted(n_time);
  
#pragma omp simd
  for (int i = 0; i < n_time; i++) {
    x_shifted(i) = x(i) - tau;
  }
  
  if (n_time > PARALLEL_THRESHOLD) {
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n_time; i++) {
      for (int j = 0; j < n_time; j++) {
        K(i, j) = std::exp(-0.5 * rho2 * std::pow(x_shifted(i) - x(j), 2));
      }
    }
  } else {
    for (int i = 0; i < n_time; i++) {
      for (int j = 0; j < n_time; j++) {
        K(i, j) = std::exp(-0.5 * rho2 * std::pow(x_shifted(i) - x(j), 2));
      }
    }
  }
  
  return beta * K;
}

// Get Sigma_y_i (covariance matrix for trial i)
// @param beta_i Trial specific amplitude
// @param K_f Covariance matrix of f
// @param Sigma_nu Covariance of AR process
// [[Rcpp::export]]
arma::mat get_Sigma_y_i(double beta_i, const arma::mat& K_f, const arma::mat& Sigma_nu) {
  return std::pow(beta_i, 2) * K_f + Sigma_nu;
}

// Get Sigma_y_i_f (conditional covariance given f) with robust error handling
// @param i trial index
// @param x time points
// @param betas amplitudes
// @param taus latencies
// @param rho GP length scale
// @param K_f Kernel matrix
// @param K_f_inv Inverse kernel matrix
// @param Sigma_nu AR covariance matrix
// [[Rcpp::export]]
arma::mat getSigma_y_i_f(int i, const arma::vec& x, const arma::vec& betas, 
                         const arma::vec& taus, double rho,
                         const arma::mat& K_f, const arma::mat& K_f_inv, 
                         const arma::mat& Sigma_nu) {
  try {
    // Get components
    arma::mat Sigma_y_i = get_Sigma_y_i(betas(i), K_f, Sigma_nu);
    arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
    
    // Calculate conditional covariance with robust matrix operations
    arma::mat Sigma_i = Sigma_y_i - K_i.t() * K_f_inv * K_i;
    
    // Ensure symmetry for numerical stability
    Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0;
    
    // Ensure positive definiteness
    Sigma_i = ensure_positive_definite(Sigma_i);
    
    return Sigma_i;
  } catch (const std::exception& e) {
    std::string error_msg = "Error in getSigma_y_i_f for trial " + std::to_string(i) + ": " + e.what();
    errorCollector.addError(error_msg);
    
    // Return fallback covariance matrix with strong regularization
    int n_time = x.n_elem;
    arma::mat fallback = arma::eye(n_time, n_time);
    return fallback;
  }
}

// Get y_hat (predicted mean for trial i) with improved error handling
// @param i trial index
// @param f structural signal
// @param betas amplitudes
// @param taus latencies
// @param rho GP length scale
// @param K_f_inv Inverse kernel matrix
// [[Rcpp::export]]
arma::vec get_y_hat(int i, const arma::vec& f, const arma::vec& betas,
                    const arma::vec& taus, double rho, const arma::mat& K_f_inv) {
  try {
    arma::vec x = arma::linspace(0, 1, f.n_elem);
    arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
    arma::vec mu = K_i * K_f_inv * f;
    return mu;
  } catch (const std::exception& e) {
    std::string error_msg = "Error in get_y_hat for trial " + std::to_string(i) + ": " + e.what();
    errorCollector.addError(error_msg);
    
    // Return zero vector as fallback
    return arma::zeros(f.n_elem);
  }
}

// Get y_hat matrix for all trials with improved parallelization
// @param y data matrix
// @param f structural signal
// @param betas amplitudes
// @param taus latencies
// @param rho GP length scale
// @param K_f_inv Inverse kernel matrix
// [[Rcpp::export]]
arma::mat get_y_hat_matrix(const arma::mat& y, const arma::vec& f, 
                           const arma::vec& betas, const arma::vec& taus,
                           double rho, const arma::mat& K_f_inv) {
  int n_time = y.n_rows;
  int n = y.n_cols;
  arma::mat y_hat(n_time, n);
  
  if (n > PARALLEL_THRESHOLD) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
      y_hat.col(i) = get_y_hat(i, f, betas, taus, rho, K_f_inv);
    }
  } else {
    for (int i = 0; i < n; i++) {
      y_hat.col(i) = get_y_hat(i, f, betas, taus, rho, K_f_inv);
    }
  }
  
  // Report any errors after parallel region
  errorCollector.reportErrors();
  
  return y_hat;
}


// Multivariate normal log density with comprehensive error handling
// @param x Vector
// @param mean Mean vector
// @param sigma Covariance matrix
// [[Rcpp::export]]
double dmvnorm(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma, bool log_p = true) {
  int n = x.n_elem;
  double log_det_sigma;
  double sign;
  arma::mat sigma_reg = sigma;
  
  // Check for non-finite values and replace them
  if (!sigma_reg.is_finite()) {
      // Replace NaN/Inf with small values
      sigma_reg.elem(arma::find_nonfinite(sigma_reg)).fill(1e-6);
      
      // If matrix is completely broken, use identity matrix
      if (!sigma_reg.is_finite()) {
          sigma_reg = arma::eye(n, n);
      }
  }
  
  // Ensure sigma is symmetric
  sigma_reg = (sigma_reg + sigma_reg.t()) / 2.0;
  
  // Try to compute log determinant
  try {
      log_det(log_det_sigma, sign, sigma_reg);
  }
  catch (const std::exception&) {
      // If log determinant calculation fails, regularize and try again
      sigma_reg.diag() += 1e-6;
      log_det(log_det_sigma, sign, sigma_reg);
  }
  
  // Avoid negative determinants (should not happen with proper regularization)
  if (sign < 0 || !std::isfinite(log_det_sigma)) {
      // Add stronger regularization
      sigma_reg.diag() += 1e-4;
      log_det(log_det_sigma, sign, sigma_reg);
  }
  
  // Calculate quadratic form with robust matrix operations
  arma::vec x_centered = x - mean;
  double quadform;
  
  try {
      // Try using inverse
      arma::mat sigma_inv = robust_inv(sigma_reg);
      quadform = arma::as_scalar(x_centered.t() * sigma_inv * x_centered);
      
      // Double-check result
      if (!std::isfinite(quadform)) {
          // Fall back to diagonal approximation
          arma::vec diag_elements = sigma_reg.diag();
          arma::vec diag_inv = 1.0 / arma::clamp(diag_elements, 1e-8, INFINITY);
          quadform = arma::sum(arma::square(x_centered) % diag_inv);
      }
  }
  catch (const std::exception&) {
      // Use more aggressive regularization and pinv as last resort
      sigma_reg.diag() += 1e-3;
      try {
          arma::mat sigma_inv = arma::pinv(sigma_reg);
          quadform = arma::as_scalar(x_centered.t() * sigma_inv * x_centered);
      } catch (...) {
          // Most extreme fallback - identity matrix approximation
          quadform = arma::sum(arma::square(x_centered));
      }
  }
  
  // Ensure quadform is positive
  quadform = std::max(quadform, 1e-10);
  
  // Calculate log PDF
  double log_pdf = -0.5 * n * std::log(2.0 * M_PI) - 0.5 * log_det_sigma - 0.5 * quadform;
  
  // Handle extreme values
  if (!std::isfinite(log_pdf)) {
      log_pdf = -1e10;  // A very negative number for log-scale
  }
  
  if (log_p) {
      return log_pdf;
  } else {
      return std::exp(std::min(log_pdf, 700.0));  // Prevent overflow
  }
}

// Prior for tau (latency) with improved stability
// @param tau latency values
// @param tau_prior_sd prior SD
// [[Rcpp::export]]
double prior_tau(const arma::vec& tau, double tau_prior_sd) {
  int n = tau.n_elem;
  
  // Create covariance matrix
  arma::mat Sigma = std::pow(tau_prior_sd, 2) * (arma::eye(n, n) - arma::ones(n, n) / (n + 1));
  
  // Add small regularization for numerical stability
  Sigma.diag() += 1e-6;
  
  // Return log density
  return dmvnorm(tau, arma::zeros(n), Sigma, true);
}

// Prior for rho (GP length scale)
// @param rho length scale value
// @param rho_prior_shape shape parameter
// @param rho_prior_scale scale parameter
// [[Rcpp::export]]
double prior_rho(double rho, double rho_prior_shape, double rho_prior_scale) {
  // Protect against invalid inputs
  if (rho <= 0 || rho_prior_shape <= 0 || rho_prior_scale <= 0) {
      return -1e10;  // Return very low log probability
  }
  
  // Gamma log density with scale parameterization
  return (rho_prior_shape - 1) * std::log(rho) - 
         rho / rho_prior_scale - 
         rho_prior_shape * std::log(rho_prior_scale) - 
         std::lgamma(rho_prior_shape);
}

// Likelihood function with robust error handling and optimization
// @param y data matrix
// @param f structural signal
// @param betas amplitudes
// @param taus latencies
// @param rho GP length scale
// @param phi AR coefficients
// @param sigma error SD
// @param K_f Kernel matrix
// @param K_f_inv Inverse kernel matrix
// [[Rcpp::export]]
double likelihood(const arma::mat& y, const arma::vec& f, 
                  const arma::vec& betas, const arma::vec& taus,
                  double rho, const arma::vec& phi, double sigma,
                  const arma::mat& K_f, const arma::mat& K_f_inv) {
  int n = y.n_cols;
  int n_time = y.n_rows;
  
  arma::vec x = arma::linspace(0, 1, n_time);
  arma::mat Sigma_nu;
  
  try {
    Sigma_nu = get_Sigma_nu(phi, sigma, n_time);
  }
  catch (const std::exception& e) {
    errorCollector.addError(std::string("Error in get_Sigma_nu: ") + e.what());
    
    // Use diagonal matrix as fallback
    Sigma_nu = std::pow(sigma, 2) * arma::eye(n_time, n_time);
  }
  
  // Thread-local accumulators for parallel reduction
  std::vector<double> local_results;
  
  if (n > PARALLEL_THRESHOLD) {
    local_results.resize(omp_get_max_threads(), 0.0);
    
#pragma omp parallel
{
  int thread_id = omp_get_thread_num();
  double thread_result = 0.0;
  
#pragma omp for schedule(dynamic)
  for (int i = 0; i < n; i++) {
    try {
      arma::mat Sigma_y_i_f = getSigma_y_i_f(i, x, betas, taus, rho, K_f, K_f_inv, Sigma_nu);
      arma::vec mu = get_y_hat(i, f, betas, taus, rho, K_f_inv);
      double trial_ll = dmvnorm(y.col(i), mu, Sigma_y_i_f, true);
      
      // Only add finite values
      if (std::isfinite(trial_ll)) {
        thread_result += trial_ll;
      } else {
        thread_result -= 100.0;  // Small penalty for non-finite likelihood
      }
    }
    catch (const std::exception& e) {
      // Store error in thread-safe way
      std::string error_msg = "Error in likelihood for trial " + std::to_string(i) + ": " + e.what();
      errorCollector.addError(error_msg);
      
      // Apply small penalty
      thread_result -= 100.0;
    }
  }
  
  // Save thread result
  local_results[thread_id] = thread_result;
}

// Combine results from all threads
double result = 0.0;
for (double local_result : local_results) {
  result += local_result;
}

// Report any errors after parallel region
errorCollector.reportErrors();

// Ensure result is finite
if (!std::isfinite(result)) {
  return -1e8;  // Less extreme value
}

return result;
  } else {
    // Sequential version for small n
    double result = 0.0;
    
    for (int i = 0; i < n; i++) {
      try {
        arma::mat Sigma_y_i_f = getSigma_y_i_f(i, x, betas, taus, rho, K_f, K_f_inv, Sigma_nu);
        arma::vec mu = get_y_hat(i, f, betas, taus, rho, K_f_inv);
        double trial_ll = dmvnorm(y.col(i), mu, Sigma_y_i_f, true);
        
        // Only add finite values
        if (std::isfinite(trial_ll)) {
          result += trial_ll;
        } else {
          result -= 100.0;  // Small penalty for non-finite likelihood
        }
      }
      catch (const std::exception& e) {
        std::string error_msg = "Error in likelihood for trial " + std::to_string(i) + ": " + e.what();
        Rcout << error_msg << "\n";
        
        // Apply small penalty
        result -= 100.0;
      }
    }
    
    // Ensure result is finite
    if (!std::isfinite(result)) {
      return -1e8;  // Less extreme value
    }
    
    return result;
  }
}

// Proposal for tau (latency) with bounds checking
// @param tau current tau values
// @param tau_proposal_sd SD for proposal
// [[Rcpp::export]]
arma::vec propose_tau(const arma::vec& tau, double tau_proposal_sd) {
  int n = tau.n_elem;
  arma::vec proposal(n);
  
  // Create covariance matrix for proposal
  arma::mat Sigma = std::pow(tau_proposal_sd, 2) * arma::eye(n, n);
  
  // Sample from multivariate normal
  proposal = tau + arma::mvnrnd(arma::zeros(n), Sigma);
  
  // Constrain proposals to reasonable range
  for (int i = 0; i < n; i++) {
    proposal(i) = std::min(std::max(proposal(i), -0.2), 0.2);
  }
  
  return proposal;
}

// Proposal for rho (GP length scale) with bounds checking
// @param rho current rho value
// @param rho_proposal_sd SD for proposal
// [[Rcpp::export]]
double propose_rho(double rho, double rho_proposal_sd) {
  // Sample from normal distribution
  double proposal = rho + R::rnorm(0, rho_proposal_sd);
  
  // Ensure positivity and reasonable range
  return std::min(std::max(proposal, 0.1), 100.0);
}

// Check if AR coefficients represent a stationary process
// @param phi AR coefficients
// [[Rcpp::export]]
bool is_stationary(const arma::vec& phi) {
  int p = phi.n_elem;
  
  // For AR(1), just check |phi| < 1
  if (p == 1) {
    return std::abs(phi(0)) < 1.0;
  }
  // For AR(2), check conditions directly
  else if (p == 2) {
    double phi1 = phi(0);
    double phi2 = phi(1);
    
    // Stationarity conditions for AR(2)
    bool cond1 = phi2 + phi1 < 1.0;
    bool cond2 = phi2 - phi1 < 1.0;
    bool cond3 = std::abs(phi2) < 1.0;
    
    return cond1 && cond2 && cond3;
  }
  // For AR(p), create companion matrix and check eigenvalues
  else {
    arma::mat A = arma::zeros(p, p);
    
    // Fill first row with coefficients
    for (int i = 0; i < p; i++) {
      A(0, i) = phi(i);
    }
    
    // Fill subdiagonal with 1's
    for (int i = 1; i < p; i++) {
      A(i, i-1) = 1.0;
    }
    
    // Get eigenvalues
    arma::cx_vec eigval;
    try {
      eigval = arma::eig_gen(A);
    }
    catch (const std::exception& e) {
      errorCollector.addError(std::string("Error in eigenvalue calculation: ") + e.what());
      return false;
    }
    
    // Check if all eigenvalues have magnitude < 1
    for (int i = 0; i < p; i++) {
      if (std::abs(eigval(i)) >= 1.0) {
        return false;
      }
    }
    
    return true;
  }
}

// Sample from AR posterior with optimized algorithm
// @param z residuals matrix
// @param ar_order AR order
// [[Rcpp::export]]
List sample_AR(const arma::mat& z, int ar_order) {
  int n_time = z.n_rows;
  int n = z.n_cols;
  
  // Flatten z into a single vector
  arma::vec z_flat = arma::vectorise(z);
  int n_total = z_flat.n_elem;
  
  // Create design matrix X and response vector y
  int n_obs = n_total - ar_order;
  arma::mat X(n_obs, ar_order, arma::fill::zeros);
  arma::vec y(n_obs, arma::fill::zeros);
  
  // Efficient construction of design matrix
  int row_idx = 0;
  for (int trial = 0; trial < n; trial++) {
    for (int t = ar_order; t < n_time; t++) {
      for (int j = 0; j < ar_order; j++) {
        X(row_idx, j) = z(t - j - 1, trial);
      }
      y(row_idx) = z(t, trial);
      row_idx++;
    }
  }
  
  // OLS estimation with robust error handling
  arma::vec beta_hat;
  try {
    // Calculate X'X and X'y efficiently
    arma::mat XtX = X.t() * X;
    arma::vec Xty = X.t() * y;
    
    // Add small regularization for numerical stability
    XtX.diag() += 1e-6;
    
    // Solve linear system
    beta_hat = arma::solve(XtX, Xty);
  }
  catch (const std::exception& e) {
    errorCollector.addError(std::string("Error in AR coefficient estimation: ") + e.what());
    
    // Return conservative defaults
    beta_hat = arma::zeros(ar_order);
    if (ar_order >= 1) beta_hat(0) = 0.1; // Conservative AR(1) coefficient
    if (ar_order >= 2) beta_hat(1) = 0.05; // Conservative AR(2) coefficient
  }
  
  // Calculate residuals and estimate sigma²
  arma::vec residuals = y - X * beta_hat;
  double sigma2 = arma::dot(residuals, residuals) / (n_obs - ar_order);
  
  // Enforce minimum value for sigma²
  sigma2 = std::max(sigma2, 1e-4);
  
  // Calculate posterior covariance matrix for AR coefficients
  arma::mat Sigma_beta;
  try {
    arma::mat XtX = X.t() * X;
    XtX.diag() += 1e-6;  // Regularization
    Sigma_beta = sigma2 * arma::inv_sympd(XtX);
  }
  catch (const std::exception& e) {
    errorCollector.addError(std::string("Error in AR coefficient covariance calculation: ") + e.what());
    
    // Use diagonal covariance as fallback
    Sigma_beta = sigma2 * arma::eye(ar_order, ar_order) * 0.01;
  }
  
  // Sample AR coefficients with stationarity constraint
  int max_attempts = 100;  // Reduced from 1000 for efficiency
  int attempt = 0;
  arma::vec phi_sample;
  bool stationary = false;
  
  while (!stationary && attempt < max_attempts) {
    try {
      phi_sample = arma::mvnrnd(beta_hat, Sigma_beta);
      stationary = is_stationary(phi_sample);
    }
    catch (const std::exception& e) {
      errorCollector.addError(std::string("Error in AR coefficient sampling: ") + e.what());
      
      // Add small noise to OLS estimate
      phi_sample = beta_hat + 0.01 * arma::randn(ar_order);
      stationary = is_stationary(phi_sample);
    }
    
    attempt++;
  }
  
  // If no stationary sample found, return shrunk coefficients
  if (!stationary) {
    phi_sample = 0.8 * beta_hat;
    
    // Force stationarity for AR(1) and AR(2)
    if (ar_order == 1) {
      phi_sample(0) = std::min(std::abs(phi_sample(0)), 0.9) * 
        (phi_sample(0) >= 0 ? 1 : -1);
    }
    else if (ar_order == 2) {
      // Scale both coefficients to ensure stationarity
      double scale = 0.9;
      while (!is_stationary(scale * phi_sample) && scale > 0.1) {
        scale -= 0.1;
      }
      phi_sample = scale * phi_sample;
    }
  }
  
  return List::create(
    Named("rho") = phi_sample,
    Named("sigma2") = sigma2
  );
}

// Sample f (structural signal) with optimized matrix operations
// @param y data matrix
// @param betas amplitudes
// @param taus latencies
// @param rho GP length scale
// @param phi AR coefficients
// @param sigma error SD
// @param n_draws number of samples to draw
// @param nugget small value for numerical stability
// [[Rcpp::export]]
arma::mat sample_f(const arma::mat& y, const arma::vec& betas, 
                   const arma::vec& taus, double rho, 
                   const arma::vec& phi, double sigma,
                   int n_draws = 1, double nugget = NUMERICAL_STABILITY_NUGGET) {
  int n_time = y.n_rows;
  int n = y.n_cols;
  arma::mat f_draws(n_time, n_draws);
  
  // Precompute common values
  arma::vec x = arma::linspace(0, 1, n_time);
  arma::mat K_f = sq_exp_kernel(x, rho, nugget);
  arma::mat K_f_inv;
  
  try {
    K_f_inv = robust_inv(K_f, nugget);
  }
  catch (const std::exception& e) {
    errorCollector.addError(std::string("Error in K_f inversion: ") + e.what());
    
    // Fallback with strong regularization
    K_f.diag() += 1e-3;
    K_f_inv = robust_inv(K_f, 1e-3);
  }
  
  // Get AR covariance matrix
  arma::mat Sigma_nu;
  try {
    Sigma_nu = get_Sigma_nu(phi, sigma, n_time);
  }
  catch (const std::exception& e) {
    errorCollector.addError(std::string("Error in get_Sigma_nu: ") + e.what());
    
    // Use diagonal matrix as fallback
    Sigma_nu = std::pow(sigma, 2) * arma::eye(n_time, n_time);
  }
  
  // Report any errors
  errorCollector.reportErrors();
  
  // For each requested draw
  for (int iter = 0; iter < n_draws; iter++) {
    // Progress reporting (every 10 iterations)
    if (iter % 10 == 0 && iter > 0) {
      Rcout << iter / 10;
    }
    
    // Initialize matrices for posterior calculation
    arma::mat A = K_f_inv;
    arma::vec b = arma::zeros(n_time);
    int successful_trials = 0;
    
    // Process each trial
    for (int i = 0; i < n; i++) {
      try {
        // Get trial-specific matrices
        arma::mat Sigma_y_i = get_Sigma_y_i(betas(i), K_f, Sigma_nu);
        arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
        
        // Calculate Sigma_i with careful regularization
        arma::mat Sigma_i = Sigma_y_i - K_i.t() * K_f_inv * K_i;
        Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0;  // Enforce symmetry
        
        // Ensure positive-definiteness
        Sigma_i = ensure_positive_definite(Sigma_i);
        
        // Calculate inverse with robust method
        arma::mat Sigma_i_inv = robust_inv(Sigma_i);
        
        // Check for extreme values before proceeding
        if (arma::any(arma::vectorise(arma::abs(Sigma_i_inv) > 1e8))) {
          throw std::runtime_error("Extreme values in Sigma_i_inv");
        }
        
        arma::mat L = K_i * K_f_inv;
        arma::mat G = Sigma_i_inv * L;
        
        // Update A and b
        A = A + L.t() * G;
        b = b + L.t() * Sigma_i_inv * y.col(i);
        successful_trials++;
      }
      catch (const std::exception& e) {
        std::string error_msg = "Error in trial " + std::to_string(i) + " sampling f: " + e.what();
        errorCollector.addError(error_msg);
        // Skip this trial
      }
    }
    
    // Skip this iteration if too few trials were successful
    if (successful_trials < n / 4) {
      // Return a simple smoothed version of the data mean as fallback
      arma::vec y_mean = arma::mean(y, 1);
      arma::vec smooth_y_mean = y_mean;
      
      // Apply simple smoothing filter
      for (int i = 2; i < n_time-2; i++) {
        smooth_y_mean(i) = (y_mean(i-2) + 2*y_mean(i-1) + 3*y_mean(i) + 
          2*y_mean(i+1) + y_mean(i+2)) / 9.0;
      }
      
      f_draws.col(iter) = smooth_y_mean;
      continue;
    }
    
    // Ensure A is symmetric
    A = (A + A.t()) / 2.0;
    
    // Apply regularization to A for numerical stability
    A.diag() += 1e-4;
    
    // Calculate posterior covariance and mean
    arma::mat K_f_post;
    arma::vec mean_f;
    
    try {
      // Try using robust inverse
      K_f_post = robust_inv(A);
      
      // Ensure symmetry
      K_f_post = (K_f_post + K_f_post.t()) / 2.0;
      
      // Calculate posterior mean
      mean_f = K_f_post * b;
      
      // Check for extreme values or NaN in mean_f
      if (!arma::is_finite(mean_f)) {
        throw std::runtime_error("Non-finite values in posterior mean");
      }
    }
    catch (const std::exception& e) {
      errorCollector.addError(std::string("Error in K_f_post calculation: ") + e.what());
      
      // Fall back to data mean
      mean_f = arma::mean(y, 1);
      
      // Create diagonal covariance
      K_f_post = arma::eye(n_time, n_time) * 0.01;
    }
    
    // Sample from multivariate normal using multiple fallback methods
    bool sampling_success = false;
    
    // Method 1: Try Cholesky
    try {
      arma::mat chol_K_f_post = arma::chol(K_f_post, "lower");
      arma::vec z = arma::randn(n_time);
      f_draws.col(iter) = mean_f + chol_K_f_post * z;
      sampling_success = true;
    }
    catch (const std::exception& e) {
      errorCollector.addError(std::string("Cholesky sampling failed: ") + e.what());
    }
    
    // Method 2: Try eigendecomposition
    if (!sampling_success) {
      try {
        arma::vec eigval;
        arma::mat eigvec;
        arma::eig_sym(eigval, eigvec, K_f_post);
        
        // Ensure positive eigenvalues
        for (unsigned int i = 0; i < eigval.n_elem; i++) {
          eigval(i) = std::max(eigval(i), 1e-5);
        }
        
        // Create matrix square root
        arma::mat sqrt_mat = eigvec * arma::diagmat(arma::sqrt(eigval)) * eigvec.t();
        
        // Generate sample
        arma::vec z = arma::randn(n_time);
        f_draws.col(iter) = mean_f + sqrt_mat * z;
        sampling_success = true;
      }
      catch (const std::exception& e) {
        errorCollector.addError(std::string("Eigendecomposition sampling failed: ") + e.what());
      }
    }
    
    // Method 3: Use mean plus small noise as fallback
    if (!sampling_success) {
      arma::vec small_noise = arma::randn(n_time) * 1e-3;
      f_draws.col(iter) = mean_f + small_noise;
    }
  }
  
  // Report any errors
  errorCollector.reportErrors();
  
  return f_draws;
}

// Sample beta (amplitude) using normal prior
// @param y data matrix
// @param y_hat predicted values
// @param beta_prior_mu prior mean
// @param beta_prior_sd prior SD
// [[Rcpp::export]]
arma::vec sample_beta(const arma::mat& y, const arma::mat& y_hat,
                      double beta_prior_mu, double beta_prior_sd) {
  int n = y.n_cols;
  arma::vec betas(n);
  
  if (n > PARALLEL_THRESHOLD) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
      // Extract vectors for this trial
      arma::vec y_i = y.col(i);
      arma::vec y_hat_i = y_hat.col(i);
      
      // Calculate sufficient statistics
      double ss_y_hat = arma::dot(y_hat_i, y_hat_i);
      double ss_y_y_hat = arma::dot(y_i, y_hat_i);
      
      // Handle near-zero predictor variance
      if (ss_y_hat < 1e-8) {
        betas(i) = beta_prior_mu;
        continue;
      }
      
      // Calculate posterior parameters
      double var_prior = std::pow(beta_prior_sd, 2);
      double var_post = 1.0 / (1.0 / var_prior + ss_y_hat);
      double mean_post = var_post * (beta_prior_mu / var_prior + ss_y_y_hat);
      
      // Sample from posterior
      double beta_sample = R::rnorm(mean_post, std::sqrt(var_post));
      
      // Constrain to reasonable range
      betas(i) = std::min(std::max(beta_sample, 0.1), 5.0);
    }
  } else {
    for (int i = 0; i < n; i++) {
      arma::vec y_i = y.col(i);
      arma::vec y_hat_i = y_hat.col(i);
      
      // Calculate sufficient statistics
      double ss_y_hat = arma::dot(y_hat_i, y_hat_i);
      double ss_y_y_hat = arma::dot(y_i, y_hat_i);
      
      // Handle near-zero predictor variance
      if (ss_y_hat < 1e-8) {
        betas(i) = beta_prior_mu;
        continue;
      }
      
      // Calculate posterior parameters
      double var_prior = std::pow(beta_prior_sd, 2);
      double var_post = 1.0 / (1.0 / var_prior + ss_y_hat);
      double mean_post = var_post * (beta_prior_mu / var_prior + ss_y_y_hat);
      
      // Sample from posterior
      double beta_sample = R::rnorm(mean_post, std::sqrt(var_post));
      
      // Constrain to reasonable range
      betas(i) = std::min(std::max(beta_sample, 0.1), 5.0);
    }
  }
  
  return betas;
}

// Sample tau (latency) using Metropolis step
// @param y data matrix
// @param f structural signal
// @param betas amplitudes
// @param taus current latencies
// @param rho GP length scale
// @param phi AR coefficients
// @param sigma error SD
// @param tau_prior_sd prior SD
// @param tau_proposal_sd proposal SD
// [[Rcpp::export]]
arma::vec sample_tau(const arma::mat& y, const arma::vec& f,
                     const arma::vec& betas, const arma::vec& taus,
                     double rho, const arma::vec& phi, double sigma,
                     double tau_prior_sd, double tau_proposal_sd) {
  int n_time = y.n_rows;
  arma::vec x = arma::linspace(0, 1, n_time);
  
  // Compute kernel matrices once
  arma::mat K_f = sq_exp_kernel(x, rho, NUMERICAL_STABILITY_NUGGET);
  arma::mat K_f_inv = robust_inv(K_f);
  
  // Propose new taus
  arma::vec taus_proposed = propose_tau(taus, tau_proposal_sd);
  
  // Calculate likelihoods and priors
  double lik_current = likelihood(y, f, betas, taus, rho, phi, sigma, K_f, K_f_inv);
  double prior_current = prior_tau(taus, tau_prior_sd);
  
  double lik_proposed = likelihood(y, f, betas, taus_proposed, rho, phi, sigma, K_f, K_f_inv);
  double prior_proposed = prior_tau(taus_proposed, tau_prior_sd);
  
  // Calculate acceptance probability
  double log_ratio = lik_proposed + prior_proposed - lik_current - prior_current;
  
  // Handle numerical issues by clamping log_ratio
  log_ratio = std::min(std::max(log_ratio, -50.0), 50.0);
  double prob = std::exp(log_ratio);
  
  // Accept or reject
  if (R::runif(0, 1) < prob) {
    return taus_proposed;
  } else {
    return taus;
  }
}

// Sample rho (GP length scale) using Metropolis step
// @param y data matrix
// @param f structural signal
// @param betas amplitudes
// @param taus latencies
// @param rho current rho
// @param phi AR coefficients
// @param sigma error SD
// @param rho_prior_shape prior shape
// @param rho_prior_scale prior scale
// @param rho_proposal_sd proposal SD
// [[Rcpp::export]]
double sample_rho(const arma::mat& y, const arma::vec& f,
                  const arma::vec& betas, const arma::vec& taus,
                  double rho, const arma::vec& phi, double sigma,
                  double rho_prior_shape, double rho_prior_scale,
                  double rho_proposal_sd) {
  int n_time = y.n_rows;
  arma::vec x = arma::linspace(0, 1, n_time);
  
  // Propose new rho
  double rho_proposed = propose_rho(rho, rho_proposal_sd);
  
  // Skip if proposed value too close to current
  if (std::abs(rho_proposed - rho) < 1e-5) {
    return rho;
  }
  
  // Compute kernel matrices
  arma::mat K_f_curr = sq_exp_kernel(x, rho, NUMERICAL_STABILITY_NUGGET);
  arma::mat K_f_curr_inv = robust_inv(K_f_curr);
  
  arma::mat K_f_prop = sq_exp_kernel(x, rho_proposed, NUMERICAL_STABILITY_NUGGET);
  arma::mat K_f_prop_inv = robust_inv(K_f_prop);
  
  // Calculate likelihoods and priors
  double lik_current = likelihood(y, f, betas, taus, rho, phi, sigma, K_f_curr, K_f_curr_inv);
  double prior_current = prior_rho(rho, rho_prior_shape, rho_prior_scale);
  
  double lik_proposed = likelihood(y, f, betas, taus, rho_proposed, phi, sigma, K_f_prop, K_f_prop_inv);
  double prior_proposed = prior_rho(rho_proposed, rho_prior_shape, rho_prior_scale);
  
  // Calculate acceptance probability with clamping
  double log_ratio = lik_proposed + prior_proposed - lik_current - prior_current;
  log_ratio = std::min(std::max(log_ratio, -50.0), 50.0);
  double prob = std::exp(log_ratio);
  
  // Accept or reject
  if (R::runif(0, 1) < prob) {
    return rho_proposed;
  } else {
    return rho;
  }
}

// Main function to fit the RPAGP model
// [[Rcpp::export]]
List fit_rpagp_cpp(const arma::mat& y, int n_iter,
                   const List& theta0,
                   const List& hyperparam,
                   int pinned_point, double pinned_value = 1.0) {
  // Set up OpenMP environment
  configureParallelEnvironment();
  
  // Start timing
  auto start = std::chrono::high_resolution_clock::now();
  
  // Extract dimensions
  int n_time = y.n_rows;
  int n = y.n_cols;
  arma::vec x = arma::linspace(0, 1, n_time);
  
  // Extract initial values with validation
  double rho0 = as<double>(theta0["rho"]);
  rho0 = std::min(std::max(rho0, 0.1), 100.0);  // Constrain to reasonable range
  
  arma::vec beta0 = as<arma::vec>(theta0["beta"]);
  for (unsigned int i = 0; i < beta0.n_elem; i++) {
    beta0(i) = std::min(std::max(beta0(i), 0.1), 5.0);  // Constrain to reasonable range
  }
  
  arma::vec tau0 = as<arma::vec>(theta0["tau"]);
  for (unsigned int i = 0; i < tau0.n_elem; i++) {
    tau0(i) = std::min(std::max(tau0(i), -0.2), 0.2);  // Constrain to reasonable range
  }
  
  arma::vec phi0 = as<arma::vec>(theta0["phi"]);
  // Ensure stationarity of initial AR parameters
  if (!is_stationary(phi0)) {
    if (phi0.n_elem == 1) {
      phi0(0) = 0.5;  // Conservative AR(1) coefficient
    } else if (phi0.n_elem == 2) {
      phi0(0) = 0.5;  // Conservative AR(2) coefficients
      phi0(1) = 0.1;
    }
  }
  
  double sigma0 = as<double>(theta0["sigma"]);
  sigma0 = std::max(sigma0, 0.01);  // Ensure positive
  
  // Check for valid AR parameter order
  int p = phi0.n_elem;
  Rcout << "AR process order: " << p << "\n";
  
  // Extract hyperparameters
  double tau_prior_sd = as<double>(hyperparam["tau_prior_sd"]);
  double tau_proposal_sd = as<double>(hyperparam["tau_proposal_sd"]);
  double rho_prior_shape = as<double>(hyperparam["rho_prior_shape"]);
  double rho_prior_scale = as<double>(hyperparam["rho_prior_scale"]);
  double rho_proposal_sd = as<double>(hyperparam["rho_proposal_sd"]);
  double beta_prior_mu = as<double>(hyperparam["beta_prior_mu"]);
  double beta_prior_sd = as<double>(hyperparam["beta_prior_sd"]);
  
  // Validate pinned_point
  if (pinned_point < 0 || pinned_point >= n_time) {
    Rcout << "Warning: pinned_point out of bounds, setting to middle point\n";
    pinned_point = n_time / 2;
  }
  
  // Initialize result containers
  Rcpp::List chain(n_iter);
  Rcpp::List chain_f(n_iter);
  Rcpp::List chain_y_hat(n_iter);
  Rcpp::List chain_z(n_iter);
  
  // Set initial values in first element
  chain[0] = theta0;
  
  // Compute initial kernel with increased stability
  arma::mat K_f = sq_exp_kernel(x, rho0, NUMERICAL_STABILITY_NUGGET);
  arma::mat K_f_inv = robust_inv(K_f);
  
  // Sample initial f with fallback options
  Rcout << "Sampling initial f...\n";
  arma::mat f_init;
  try {
    f_init = sample_f(y, beta0, tau0, rho0, phi0, sigma0, 1, NUMERICAL_STABILITY_NUGGET);
  }
  catch (const std::exception& e) {
    Rcout << "Error sampling initial f: " << e.what() << "\n";
    
    // Generate a smooth curve as fallback
    f_init = arma::zeros(n_time, 1);
    
    // Start with mean of data
    arma::vec y_mean = arma::mean(y, 1);
    
    // Apply smoothing
    arma::vec smooth_mean = y_mean;
    for (int i = 2; i < n_time-2; i++) {
      smooth_mean(i) = (y_mean(i-2) + 2*y_mean(i-1) + 3*y_mean(i) + 
        2*y_mean(i+1) + y_mean(i+2)) / 9.0;
    }
    
    f_init.col(0) = smooth_mean;
  }
  
  arma::vec f = f_init.col(0);
  
  // Apply pinning with protection against division by zero
  if (std::abs(f(pinned_point)) > 1e-10) {
    f = pinned_value * f / f(pinned_point);
  } else {
    f = f + (pinned_value - f(pinned_point));
  }
  
  chain_f[0] = f;
  
  Rcout << "Starting MCMC iterations...\n";
  
  // Main MCMC loop
  for (int iter = 1; iter < n_iter; iter++) {
    // Progress reporting
    if ((iter % (n_iter/10)) == 0) {
      Rcout << " ... " << static_cast<int>((iter/static_cast<double>(n_iter))*100) << "% \n";
    }
    
    try {
      // Get current parameters
      Rcpp::List current = chain[iter-1];
      double rho = as<double>(current["rho"]);
      arma::vec beta = as<arma::vec>(current["beta"]);
      arma::vec tau = as<arma::vec>(current["tau"]);
      arma::vec phi = as<arma::vec>(current["phi"]);
      double sigma = as<double>(current["sigma"]);
      
      // Sample f and rescale
      arma::mat f_sample = sample_f(y, beta, tau, rho, phi, sigma, 1);
      f = f_sample.col(0);
      
      // Pinning with protection
      if (std::abs(f(pinned_point)) > 1e-10) {
        f = pinned_value * f / f(pinned_point);
      } else {
        f = f + (pinned_value - f(pinned_point));
      }
      
      // Update kernel and its inverse for stability
      K_f = sq_exp_kernel(x, rho, NUMERICAL_STABILITY_NUGGET);
      K_f_inv = robust_inv(K_f);
      
      // Update y_hat
      arma::mat y_hat = get_y_hat_matrix(y, f, beta, tau, rho, K_f_inv);
      
      // Sample betas
      beta = sample_beta(y, y_hat, beta_prior_mu, beta_prior_sd);
      
      // Sample f again with new betas
      f_sample = sample_f(y, beta, tau, rho, phi, sigma, 1);
      f = f_sample.col(0);
      
      // Pinning with protection
      if (std::abs(f(pinned_point)) > 1e-10) {
        f = pinned_value * f / f(pinned_point);
      } else {
        f = f + (pinned_value - f(pinned_point));
      }
      
      // Sample taus
      tau = sample_tau(y, f, beta, tau, rho, phi, sigma, tau_prior_sd, tau_proposal_sd);
      
      // Sample rho
      rho = sample_rho(y, f, beta, tau, rho, phi, sigma, rho_prior_shape, rho_prior_scale, rho_proposal_sd);
      
      // Update kernel and its inverse
      K_f = sq_exp_kernel(x, rho, NUMERICAL_STABILITY_NUGGET);
      K_f_inv = robust_inv(K_f);
      
      // Update y_hat
      y_hat = get_y_hat_matrix(y, f, beta, tau, rho, K_f_inv);
      
      // Compute residuals
      arma::mat z = y - y_hat;
      
      // Sample AR parameters
      List ar_post = sample_AR(z, p);
      arma::vec phi_new = as<arma::vec>(ar_post["rho"]);
      double sigma2_new = as<double>(ar_post["sigma2"]);
      
      // Ensure sigma is positive
      double sigma_new = std::sqrt(std::max(sigma2_new, 1e-6));
      
      // Update current parameters
      Rcpp::List current_new = clone(current);
      current_new["beta"] = beta;
      current_new["tau"] = tau;
      current_new["rho"] = rho;
      current_new["phi"] = phi_new;
      current_new["sigma"] = sigma_new;
      
      // Store results
      chain[iter] = current_new;
      chain_f[iter] = f;
      chain_y_hat[iter] = y_hat;
      chain_z[iter] = z;
    }
    catch (const std::exception& e) {
      Rcout << "Error in iteration " << iter << ": " << e.what() << "\n";
      
      // Continue with previous values
      chain[iter] = chain[iter-1];
      chain_f[iter] = chain_f[iter-1];
      chain_y_hat[iter] = chain_y_hat[iter-1];
      chain_z[iter] = chain_z[iter-1];
    }
  }
  
  // End timing
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  double runtime = elapsed.count();
  
  Rcout << "\nRPAGP MCMC completed in " << runtime << " seconds\n";
  
  // Return results
  return List::create(
    Named("chain") = chain,
    Named("chain_f") = chain_f,
    Named("chain_y_hat") = chain_y_hat,
    Named("chain_z") = chain_z,
    Named("runtime") = runtime
  );
}
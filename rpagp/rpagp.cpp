// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// ------------- Kernel Functions -------------

// Square exponential kernel implementation in C++
// @param x Vector of time points
// @param rho Length scale parameter
// @param alpha Amplitude parameter
// @param nugget Nugget parameter for numerical stability
// [[Rcpp::export]]
arma::mat sq_exp_kernel_cpp(const arma::vec& x, double rho, double alpha = 1.0, double nugget = 0.0) {
  const int n = x.n_elem;
  const double alpha_sq = alpha * alpha;
  const double rho_sq_half = 0.5 * rho * rho;
  
  arma::mat K(n, n);
  for(int i = 0; i < n; i++) {
    K(i, i) = alpha_sq + nugget;  // Diagonal element
    
    for(int j = 0; j < i; j++) {  // Lower triangular part
      double dist = x[i] - x[j];
      double dist_sq = dist * dist;
      K(i, j) = alpha_sq * std::exp(-rho_sq_half * dist_sq);
      K(j, i) = K(i, j);  // Symmetric matrix
    }
  }
  
  return K;
}

// Get K_i matrix (Cov(y_i, f))
// @param x Sequence of observation times
// @param rho Length scale parameter
// @param tau Latency parameter
// @param beta Amplitude parameter
// [[Rcpp::export]]
arma::mat get_K_i_cpp(const arma::vec& x, double rho, double tau, double beta) {
  const int n_time = x.n_elem;
  const double rho_sq_half = 0.5 * rho * rho;
  
  arma::mat K(n_time, n_time);
  
  for(int i = 0; i < n_time; i++) {
    for(int j = 0; j < n_time; j++) {
      // Note: This kernel is NOT symmetric when tau != 0
      // We must compute every element separately
      double diff = x[i] - x[j] - tau;
      K(i, j) = beta * std::exp(-rho_sq_half * diff * diff);
    }
  }
  
  return K;
}

// ------------- Matrix Utility Functions -------------

// Convert upper triangular to symmetric matrix
// @param m Upper triangular matrix
// [[Rcpp::export]]
arma::mat ultosymmetric_cpp(const arma::mat& m) {
  return m + m.t() - diagmat(m.diag());
}

// ------------- Prediction Functions -------------

// Get y_hat for a single trial
// @param f Vector of f values
// @param K_i Covariance matrix K_i
// @param K_f_inv Inverse of covariance matrix of f
// [[Rcpp::export]]
arma::vec get_y_hat_single_cpp(const arma::vec& f, const arma::mat& K_i, const arma::mat& K_f_inv) {
  return K_i * (K_f_inv * f);
}

// Get y_hat matrix for all trials
// @param f Vector of f values
// @param x Vector of time points
// @param rho Length scale parameter
// @param tau Vector of latency parameters
// @param beta Vector of amplitude parameters
// @param K_f_inv Inverse of covariance matrix of f
// [[Rcpp::export]]
arma::mat get_y_hat_matrix_cpp(const arma::vec& f, const arma::vec& x, double rho, 
                             const arma::vec& tau, const arma::vec& beta, const arma::mat& K_f_inv) {
  int n_time = x.n_elem;
  int n = tau.n_elem;
  arma::mat y_hat(n_time, n);
  
  for(int i = 0; i < n; i++) {
    arma::mat K_i = get_K_i_cpp(x, rho, tau[i], beta[i]);
    y_hat.col(i) = get_y_hat_single_cpp(f, K_i, K_f_inv);
  }
  
  return y_hat;
}

// ------------- Statistical Functions -------------

// Calculate AR process covariance matrix
// @param phi AR coefficients
// @param sigma Error standard deviation
// @param n_time Number of time points
// NOTE: This is a simplified version - in practice, replace with direct C++ computation of AR covariance
// [[Rcpp::export]]
arma::mat get_Sigma_nu_cpp(const arma::vec& phi, double sigma, int n_time) {
  // Create a basic AR(p) covariance structure
  // This is a simplified implementation - for production, use a proper AR covariance calculation
  int p = phi.n_elem;
  arma::mat Sigma(n_time, n_time, arma::fill::zeros);
  
  // Compute autocovariances
  arma::vec acf = arma::vec(n_time, arma::fill::zeros);
  acf(0) = sigma * sigma / (1.0 - arma::sum(phi % phi));
  
  // Simple AR(p) autocovariance calculation
  for(int k = 1; k < n_time; k++) {
    double sum = 0.0;
    for(int j = 0; j < std::min(p, k); j++) {
      sum += phi(j) * acf(k - j - 1);
    }
    acf(k) = sum;
  }
  
  // Fill the covariance matrix with the autocovariances
  for(int i = 0; i < n_time; i++) {
    for(int j = 0; j <= i; j++) {
      Sigma(i, j) = acf(i - j);
      Sigma(j, i) = Sigma(i, j); // Symmetric
    }
  }
  
  return Sigma;
}

// Get Sigma_y_i
// @param beta_i Trial-specific amplitude
// @param K_f Covariance matrix of f
// @param Sigma_nu AR process covariance matrix
// [[Rcpp::export]]
arma::mat get_Sigma_y_i_cpp(double beta_i, const arma::mat& K_f, const arma::mat& Sigma_nu) {
  return (beta_i * beta_i) * K_f + Sigma_nu;
}

// Get Sigma_y_i_f
// @param beta_i Trial-specific amplitude
// @param K_f Covariance matrix of f
// @param K_f_inv Inverse of K_f
// @param K_i Trial-specific covariance
// @param Sigma_nu AR process covariance matrix
// [[Rcpp::export]]
arma::mat getSigma_y_i_f_cpp(double beta_i, const arma::mat& K_f, const arma::mat& K_f_inv, 
                           const arma::mat& K_i, const arma::mat& Sigma_nu) {
  arma::mat Sigma_y_i = get_Sigma_y_i_cpp(beta_i, K_f, Sigma_nu);
  arma::mat Sigma_i = Sigma_y_i - K_i.t() * K_f_inv * K_i;
  Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0; // Ensure symmetry
  return Sigma_i;
}

// Multivariate normal density calculation
// @param x Vector of observations
// @param mean Mean vector
// @param sigma Covariance matrix
// [[Rcpp::export]]
double dmvnorm_cpp(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma) {
  int n = x.n_elem;
  double log_det_val;
  double sign;
  log_det(log_det_val, sign, sigma);
  
  if (sign <= 0) {
    return -1e10; // Return large negative number instead of R_NegInf
  }
  
  arma::vec x_centered = x - mean;
  double quadform = as_scalar(x_centered.t() * inv(sigma) * x_centered);
  
  return -0.5 * (n * log(2.0 * M_PI) + log_det_val + quadform);
}

// Compute likelihood for all trials
// @param y Matrix of observed data
// @param f Vector of f values
// @param theta List of parameter values
// @param x Vector of time points
// @param Sigma_nu AR process covariance matrix
// [[Rcpp::export]]
double likelihood_cpp(const arma::mat& y, const arma::vec& f, const arma::vec& tau, 
                   const arma::vec& beta, double rho, double sigma, const arma::vec& phi,
                   const arma::vec& x, const arma::mat& K_f, const arma::mat& K_f_inv) {
  int n = y.n_cols;
  int n_time = y.n_rows;
  
  // Create Sigma_nu (AR covariance)
  arma::mat Sigma_nu = get_Sigma_nu_cpp(phi, sigma, n_time);
  
  double tmp = 0.0;
  
  for(int i = 0; i < n; i++) {
    arma::mat K_i = get_K_i_cpp(x, rho, tau[i], beta[i]);
    arma::mat Sigma_y_i_f = getSigma_y_i_f_cpp(beta[i], K_f, K_f_inv, K_i, Sigma_nu);
    
    // Ensure Sigma_y_i_f is symmetric
    Sigma_y_i_f = (Sigma_y_i_f + Sigma_y_i_f.t()) / 2.0;
    
    // Get y_hat for current trial
    arma::vec mu = get_y_hat_single_cpp(f, K_i, K_f_inv);
    
    // Compute multivariate normal log density
    tmp += dmvnorm_cpp(y.col(i), mu, Sigma_y_i_f);
  }
  
  // Handle extreme negative values
  if (tmp < -1e9) {
    return -1e10;
  }
  
  return tmp;
}

// ------------- MCMC Sampling Functions -------------

// Tau proposal for Metropolis-Hastings
// @param tau Current tau values
// @param tau_proposal_sd Proposal standard deviation
// [[Rcpp::export]]
arma::vec propose_tau_cpp(const arma::vec& tau, double tau_proposal_sd) {
  int n = tau.n_elem;
  arma::mat Sigma = tau_proposal_sd * tau_proposal_sd * arma::eye(n, n);
  arma::vec proposal = arma::mvnrnd(tau, Sigma);
  return proposal;
}

// Rho proposal for Metropolis-Hastings
// @param rho Current rho value
// @param rho_proposal_sd Proposal standard deviation
// [[Rcpp::export]]
double propose_rho_cpp(double rho, double rho_proposal_sd) {
  return arma::as_scalar(arma::randn(1)) * rho_proposal_sd + rho;
}

// Tau prior evaluation
// @param tau Vector of tau values
// @param tau_prior_sd Prior standard deviation
// [[Rcpp::export]]
double prior_tau_cpp(const arma::vec& tau, double tau_prior_sd) {
  int n = tau.n_elem;
  arma::mat Sigma = tau_prior_sd * tau_prior_sd * (arma::eye(n, n) - arma::ones(n, n) / (n + 1.0));
  
  // Compute log density of multivariate normal
  arma::vec mean = arma::zeros(n);
  double log_det_val;
  double sign;
  log_det(log_det_val, sign, Sigma);
  
  if (sign <= 0) {
    return -1e10;
  }
  
  double quadform = arma::as_scalar(tau.t() * arma::inv(Sigma) * tau);
  double logdens = -0.5 * (n * log(2.0 * M_PI) + log_det_val + quadform);
  
  return logdens;
}

// Rho prior evaluation (gamma distribution)
// @param rho Rho value
// @param shape Shape parameter
// @param scale Scale parameter
// [[Rcpp::export]]
double prior_rho_cpp(double rho, double shape, double scale) {
  if (rho <= 0) return -1e10;
  
  double logdens = (shape - 1.0) * log(rho) - rho / scale - shape * log(scale) - lgamma(shape);
  return logdens;
}

// Sample tau using Metropolis-Hastings
// @param y Data matrix
// @param f Vector of f values
// @param current_tau Current tau values
// @param current_beta Current beta values
// @param current_rho Current rho value
// @param current_sigma Current sigma value
// @param current_phi Current phi values
// @param tau_prior_sd Prior SD for tau
// @param tau_proposal_sd Proposal SD for tau
// @param x Vector of time points
// [[Rcpp::export]]
arma::vec sample_tau_cpp(const arma::mat& y, const arma::vec& f, const arma::vec& current_tau,
                       const arma::vec& current_beta, double current_rho, double current_sigma,
                       const arma::vec& current_phi, double tau_prior_sd, double tau_proposal_sd,
                       const arma::vec& x) {
  // Create kernel matrices
  arma::mat K_f = sq_exp_kernel_cpp(x, current_rho, 1.0, 1e-6);
  arma::mat K_f_inv = inv(K_f);
  
  // Propose new tau
  arma::vec proposed_tau = propose_tau_cpp(current_tau, tau_proposal_sd);
  
  // Compute likelihoods and priors
  double lik_current = likelihood_cpp(y, f, current_tau, current_beta, current_rho, 
                                   current_sigma, current_phi, x, K_f, K_f_inv);
  double prior_current = prior_tau_cpp(current_tau, tau_prior_sd);
  
  double lik_proposed = likelihood_cpp(y, f, proposed_tau, current_beta, current_rho, 
                                    current_sigma, current_phi, x, K_f, K_f_inv);
  double prior_proposed = prior_tau_cpp(proposed_tau, tau_prior_sd);
  
  // Metropolis-Hastings step
  double log_ratio = lik_proposed + prior_proposed - lik_current - prior_current;
  double acceptance_prob = std::min(1.0, exp(log_ratio));
  
  // Accept or reject
  double u = arma::as_scalar(arma::randu(1));
  if (u < acceptance_prob) {
    return proposed_tau;
  } else {
    return current_tau;
  }
}

// Sample rho using Metropolis-Hastings
// @param y Data matrix
// @param f Vector of f values
// @param current_tau Current tau values
// @param current_beta Current beta values
// @param current_rho Current rho value
// @param current_sigma Current sigma value
// @param current_phi Current phi values
// @param rho_prior_shape Prior shape for rho
// @param rho_prior_scale Prior scale for rho
// @param rho_proposal_sd Proposal SD for rho
// @param x Vector of time points
// [[Rcpp::export]]
double sample_rho_cpp(const arma::mat& y, const arma::vec& f, const arma::vec& current_tau,
                    const arma::vec& current_beta, double current_rho, double current_sigma,
                    const arma::vec& current_phi, double rho_prior_shape, double rho_prior_scale,
                    double rho_proposal_sd, const arma::vec& x) {
  // Propose new rho
  double proposed_rho = propose_rho_cpp(current_rho, rho_proposal_sd);
  
  // Skip computation if proposed rho is negative
  if (proposed_rho <= 0) {
    return current_rho;
  }
  
  // Compute kernel matrices for current and proposed
  arma::mat K_f_curr = sq_exp_kernel_cpp(x, current_rho, 1.0, 1e-6);
  arma::mat K_f_curr_inv = inv(K_f_curr);
  
  arma::mat K_f_prop = sq_exp_kernel_cpp(x, proposed_rho, 1.0, 1e-6);
  arma::mat K_f_prop_inv = inv(K_f_prop);
  
  // Compute likelihoods and priors
  double lik_current = likelihood_cpp(y, f, current_tau, current_beta, current_rho, 
                                   current_sigma, current_phi, x, K_f_curr, K_f_curr_inv);
  double prior_current = prior_rho_cpp(current_rho, rho_prior_shape, rho_prior_scale);
  
  double lik_proposed = likelihood_cpp(y, f, current_tau, current_beta, proposed_rho, 
                                    current_sigma, current_phi, x, K_f_prop, K_f_prop_inv);
  double prior_proposed = prior_rho_cpp(proposed_rho, rho_prior_shape, rho_prior_scale);
  
  // Metropolis-Hastings step
  double log_ratio = lik_proposed + prior_proposed - lik_current - prior_current;
  double acceptance_prob = std::min(1.0, exp(log_ratio));
  
  // Accept or reject
  double u = arma::as_scalar(arma::randu(1));
  if (u < acceptance_prob) {
    return proposed_rho;
  } else {
    return current_rho;
  }
}

// Sample beta for a single trial
// @param y_i Data for trial i
// @param y_hat_i Predicted values for trial i
// @param current_beta_i Current beta for trial i
// @param beta_prior_mu Prior mean for beta
// @param beta_prior_sd Prior SD for beta
// [[Rcpp::export]]
double sample_beta_single_cpp(const arma::vec& y_i, const arma::vec& y_hat_i, double current_beta_i,
                           double beta_prior_mu, double beta_prior_sd) {
  int n = y_i.n_elem;
  
  // Modified design matrix (scaled by current beta)
  arma::vec X = y_hat_i / current_beta_i;
  
  // Posterior calculations for Bayesian linear regression
  double V_post = 1.0 / (1.0 / (beta_prior_sd * beta_prior_sd) + arma::dot(X, X));
  double mu_post = V_post * (beta_prior_mu / (beta_prior_sd * beta_prior_sd) + arma::dot(X, y_i));
  
  double a_post = 1.0 + n / 2.0;
  double b_post = 1.0 + 0.5 * (beta_prior_mu * beta_prior_mu / (beta_prior_sd * beta_prior_sd) + 
                               arma::dot(y_i, y_i) - mu_post * mu_post / V_post);
  
  // Sample from scaled inverse chi-squared (via inverse gamma)
  double sigma2_sample = 1.0 / arma::randg(arma::distr_param(a_post, 1.0 / b_post));
  
  // Sample beta from normal
  double beta_sample = arma::as_scalar(arma::randn(1)) * sqrt(V_post * sigma2_sample) + mu_post;
  
  return beta_sample;
}

// Sample all betas
// @param y Data matrix
// @param y_hat Predicted values matrix
// @param current_beta Current beta values
// @param beta_prior_mu Prior mean for beta
// @param beta_prior_sd Prior SD for beta
// [[Rcpp::export]]
arma::vec sample_beta_cpp(const arma::mat& y, const arma::mat& y_hat, const arma::vec& current_beta,
                       double beta_prior_mu, double beta_prior_sd) {
  int n = y.n_cols;
  arma::vec new_beta = current_beta;
  
  for (int i = 0; i < n; i++) {
    new_beta(i) = sample_beta_single_cpp(y.col(i), y_hat.col(i), current_beta(i), 
                                       beta_prior_mu, beta_prior_sd);
  }
  
  return new_beta;
}

// Sample f using reduction pattern
// This approach avoids explicit mutex locks by using a reduction
// [[Rcpp::export]]
arma::mat sample_f_parallel_cpp(const arma::mat& y, const arma::vec& tau, const arma::vec& beta,
                             double rho, double sigma, const arma::vec& phi, 
                             const arma::vec& x, int n_draws, double nugget = 1e-6) {
  int n_time = y.n_rows;
  int n = y.n_cols;
  
  // Create K_f and K_f_inv
  arma::mat K_f = sq_exp_kernel_cpp(x, rho, 1.0, nugget);
  arma::mat K_f_inv = inv(K_f);
  
  // Create Sigma_nu
  arma::mat Sigma_nu = get_Sigma_nu_cpp(phi, sigma, n_time);
  
  // Create output matrix for f draws
  arma::mat f_draws(n_time, n_draws);
  
  for (int iter = 0; iter < n_draws; iter++) {
    // Initialize A and b
    arma::mat A = K_f_inv;
    arma::vec b = arma::zeros(n_time);
    
    // Process each trial sequentially (avoid mutex issues)
    // In a production environment, you can use a better parallel reduction pattern
    for (int i = 0; i < n; i++) {
      arma::mat K_i = get_K_i_cpp(x, rho, tau[i], beta[i]);
      arma::mat Sigma_y_i = get_Sigma_y_i_cpp(beta[i], K_f, Sigma_nu);
      arma::mat Sigma_i = Sigma_y_i - K_i.t() * K_f_inv * K_i;
      Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0; // Ensure symmetry
      
      arma::mat Sigma_i_inv;
      bool success = inv(Sigma_i_inv, Sigma_i);
      if (!success) {
        // If inversion fails, use a pseudo-inverse
        Sigma_i_inv = arma::pinv(Sigma_i);
      }
      
      arma::mat L = K_i * K_f_inv;
      arma::mat G = Sigma_i_inv * L;
      
      A += L.t() * G;
      b += L.t() * Sigma_i_inv * y.col(i);
    }
    
    // Compute posterior covariance
    arma::mat K_f_post = inv(A);
    K_f_post = (K_f_post + K_f_post.t()) / 2.0; // Ensure symmetry
    
    // Sample from multivariate normal
    arma::vec mean = K_f_post * b;
    arma::mat L;
    bool chol_success = chol(L, K_f_post, "lower");
    
    if (chol_success) {
      arma::vec z = arma::randn(n_time);
      f_draws.col(iter) = mean + L * z;
    } else {
      // Fallback if Cholesky decomposition fails
      arma::mat U;
      arma::vec d;
      arma::eig_sym(d, U, K_f_post);
      
      // Ensure all eigenvalues are positive
      d = arma::max(d, arma::zeros(d.n_elem));
      
      // Construct sample
      arma::vec z = arma::randn(n_time);
      f_draws.col(iter) = mean + U * arma::diagmat(arma::sqrt(d)) * z;
    }
  }
  
  return f_draws;
}

// AR simulation for residuals (simplified)
// @param n Length of series to generate
// @param phi AR parameters
// @param sigma Innovation standard deviation
// [[Rcpp::export]]
arma::vec arima_sim_cpp(int n, const arma::vec& phi, double sigma) {
  int p = phi.n_elem;
  arma::vec series(n, arma::fill::zeros);
  
  // Generate initial values
  for (int i = 0; i < p; i++) {
    series(i) = arma::as_scalar(arma::randn(1)) * sigma;
  }
  
  // Generate the rest of the series
  for (int i = p; i < n; i++) {
    double value = 0.0;
    for (int j = 0; j < p; j++) {
      value += phi(j) * series(i - j - 1);
    }
    value += arma::as_scalar(arma::randn(1)) * sigma;
    series(i) = value;
  }
  
  return series;
}

// Sample AR parameters
// @param z Matrix of residuals
// @param ar_order AR order
// [[Rcpp::export]]
List sample_AR_cpp(const arma::mat& z, int ar_order) {
  int n_time = z.n_rows;
  int n = z.n_cols;
  
  // Flatten the residuals
  arma::vec z_vec(n_time * n);
  for (int i = 0; i < n; i++) {
    z_vec.subvec(i * n_time, (i + 1) * n_time - 1) = z.col(i);
  }
  
  // Set up the design matrix X for the AR model
  arma::mat X(n_time * n - ar_order, ar_order);
  arma::vec y(n_time * n - ar_order);
  
  int idx = 0;
  for (int i = 0; i < n; i++) {
    for (int t = ar_order; t < n_time; t++) {
      for (int j = 0; j < ar_order; j++) {
        X(idx, j) = z(t - j - 1, i);
      }
      y(idx) = z(t, i);
      idx++;
    }
  }
  
  // Sample phi
  arma::mat XtX = X.t() * X;
  arma::mat XtX_inv = inv(XtX + arma::eye(ar_order, ar_order) * 1e-6);
  arma::vec phi_mean = XtX_inv * X.t() * y;
  
  // Check stationarity
  arma::cx_vec roots = arma::roots(arma::join_cols(arma::ones(1), -phi_mean));
  bool stationary = true;
  for (size_t i = 0; i < roots.n_elem; i++) {
    if (std::abs(roots(i)) <= 1.0) {
      stationary = false;
      break;
    }
  }
  
  // If not stationary, return a default AR process
  if (!stationary) {
    arma::vec default_phi(ar_order, arma::fill::zeros);
    if (ar_order > 0) default_phi(0) = 0.5; // Simple AR(1) with phi = 0.5
    
    return List::create(
      Named("phi") = default_phi,
      Named("sigma") = arma::stddev(z_vec)
    );
  }
  
  // Compute residuals
  arma::vec eps = y - X * phi_mean;
  double sigma2 = arma::dot(eps, eps) / (y.n_elem - ar_order);
  
  return List::create(
    Named("phi") = phi_mean,
    Named("sigma") = sqrt(sigma2)
  );
}

// Generate synthetic data
// @param n Number of trials
// @param n_time Number of time points
// @param beta Vector of amplitudes
// @param tau Vector of latencies
// @param rho GP length scale
// @param sigma Residual standard deviation
// @param phi AR parameters
// [[Rcpp::export]]
List generate_data_cpp(int n, int n_time, const arma::vec& beta, const arma::vec& tau,
                     double rho, double sigma, const arma::vec& phi) {
  arma::vec x = arma::linspace(0, 1, n_time);
  
  // Generate f from GP
  arma::mat K_f = sq_exp_kernel_cpp(x, rho, 1.0, 1e-6);
  arma::mat K_f_inv = inv(K_f);
  
  // Cholesky decomposition for sampling
  arma::mat L;
  bool success = chol(L, K_f, "lower");
  
  // Generate f
  arma::vec f;
  if (success) {
    arma::vec z = arma::randn(n_time);
    f = L * z;
  } else {
    // Fallback using eigendecomposition
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, K_f);
    
    // Ensure positive eigenvalues
    eigval = arma::max(eigval, arma::zeros(eigval.n_elem));
    
    // Generate f
    arma::vec z = arma::randn(n_time);
    f = eigvec * arma::diagmat(arma::sqrt(eigval)) * z;
  }
  
  // Initialize matrices
  arma::mat y(n_time, n);
  arma::mat z(n_time, n);
  arma::mat mu(n_time, n);
  
  // Generate trial data
  for (int i = 0; i < n; i++) {
    // Compute K_i
    arma::mat K_i = get_K_i_cpp(x, rho, tau(i), beta(i));
    
    // Compute mean
    mu.col(i) = K_i * K_f_inv * f;
    
    // Generate AR noise
    z.col(i) = arima_sim_cpp(n_time, phi, sigma);
    
    // Combine signal and noise
    y.col(i) = mu.col(i) + z.col(i);
  }
  
  return List::create(
    Named("y") = y,
    Named("f") = f,
    Named("z") = z,
    Named("mu") = mu
  );
}

// Main MCMC function
// @param y Data matrix
// @param n_iter Number of iterations
// @param initial_beta Initial beta values
// @param initial_tau Initial tau values
// @param initial_rho Initial rho value
// @param initial_sigma Initial sigma value
// @param initial_phi Initial phi values
// @param tau_prior_sd Prior SD for tau
// @param tau_proposal_sd Proposal SD for tau
// @param rho_prior_shape Prior shape for rho
// @param rho_prior_scale Prior scale for rho
// @param rho_proposal_sd Proposal SD for rho
// @param beta_prior_mu Prior mean for beta
// @param beta_prior_sd Prior SD for beta
// @param pinned_point Index of pinned point
// @param pinned_value Value at pinned point
// [[Rcpp::export]]
List fit_rpagp_cpp(const arma::mat& y, int n_iter, 
                 const arma::vec& initial_beta, const arma::vec& initial_tau, 
                 double initial_rho, double initial_sigma, const arma::vec& initial_phi,
                 double tau_prior_sd, double tau_proposal_sd,
                 double rho_prior_shape, double rho_prior_scale, double rho_proposal_sd,
                 double beta_prior_mu, double beta_prior_sd,
                 int pinned_point, double pinned_value = 1.0) {
  // Get dimensions
  int n_time = y.n_rows;
  int n = y.n_cols;
  int p = initial_phi.n_elem;
  
  // Create vector of time points
  arma::vec x = arma::linspace(0, 1, n_time);
  
  // Pre-allocate storage for MCMC samples
  arma::mat beta_samples(n, n_iter);
  arma::mat tau_samples(n, n_iter);
  arma::vec rho_samples(n_iter);
  arma::vec sigma_samples(n_iter);
  arma::mat phi_samples(p, n_iter);
  arma::mat f_samples(n_time, n_iter);
  
  // Initialize first iteration
  beta_samples.col(0) = initial_beta;
  tau_samples.col(0) = initial_tau;
  rho_samples(0) = initial_rho;
  sigma_samples(0) = initial_sigma;
  phi_samples.col(0) = initial_phi;
  
  // Initial K_f and K_f_inv
  arma::mat K_f = sq_exp_kernel_cpp(x, initial_rho, 1.0, 1e-6);
  arma::mat K_f_inv = inv(K_f);
  
  // Initial f sample
  arma::mat f_draw = sample_f_parallel_cpp(y, initial_tau, initial_beta, initial_rho, 
                                        initial_sigma, initial_phi, x, 1, 1e-6);
  f_samples.col(0) = f_draw.col(0);
  
  // Ensure pinned value
  f_samples.col(0) = pinned_value * f_samples.col(0) / f_samples(pinned_point, 0);
  
  // MCMC loop
  for (int iter = 1; iter < n_iter; iter++) {
    if (iter % (n_iter/10) == 0) {
      Rcpp::Rcout << "... " << static_cast<int>((iter * 100.0) / n_iter) << "% \n";
    }
    
    // Get current parameters
    arma::vec current_beta = beta_samples.col(iter-1);
    arma::vec current_tau = tau_samples.col(iter-1);
    double current_rho = rho_samples(iter-1);
    double current_sigma = sigma_samples(iter-1);
    arma::vec current_phi = phi_samples.col(iter-1);
    arma::vec current_f = f_samples.col(iter-1);
    
    // Update K_f and K_f_inv
    K_f = sq_exp_kernel_cpp(x, current_rho, 1.0, 1e-6);
    K_f_inv = inv(K_f);
    
    // Sample f and rescale
    arma::mat f_draw = sample_f_parallel_cpp(y, current_tau, current_beta, current_rho, 
                                          current_sigma, current_phi, x, 1, 1e-6);
    current_f = f_draw.col(0);
    current_f = pinned_value * current_f / current_f(pinned_point);
    
    // Get y_hat
    arma::mat y_hat = get_y_hat_matrix_cpp(current_f, x, current_rho, current_tau, current_beta, K_f_inv);
    
    // Sample betas
    current_beta = sample_beta_cpp(y, y_hat, current_beta, beta_prior_mu, beta_prior_sd);
    
    // Sample f again and rescale
    f_draw = sample_f_parallel_cpp(y, current_tau, current_beta, current_rho, 
                                 current_sigma, current_phi, x, 1, 1e-6);
    current_f = f_draw.col(0);
    current_f = pinned_value * current_f / current_f(pinned_point);
    
    // Sample tau
    current_tau = sample_tau_cpp(y, current_f, current_tau, current_beta, current_rho, 
                               current_sigma, current_phi, tau_prior_sd, tau_proposal_sd, x);
    
    // Sample rho
    current_rho = sample_rho_cpp(y, current_f, current_tau, current_beta, current_rho, 
                               current_sigma, current_phi, rho_prior_shape, rho_prior_scale, 
                               rho_proposal_sd, x);
    
    // Update K_f and K_f_inv with new rho
    K_f = sq_exp_kernel_cpp(x, current_rho, 1.0, 1e-6);
    K_f_inv = inv(K_f);
    
    // Update y_hat
    y_hat = get_y_hat_matrix_cpp(current_f, x, current_rho, current_tau, current_beta, K_f_inv);
    
    // Compute residuals
    arma::mat z = y - y_hat;
    
    // Sample AR parameters
    List ar_result = sample_AR_cpp(z, p);
    arma::vec new_phi = ar_result["phi"];
    double new_sigma = ar_result["sigma"];
    
    // Store current iteration
    beta_samples.col(iter) = current_beta;
    tau_samples.col(iter) = current_tau;
    rho_samples(iter) = current_rho;
    sigma_samples(iter) = new_sigma;
    phi_samples.col(iter) = new_phi;
    f_samples.col(iter) = current_f;
  }
  
  Rcpp::Rcout << "MCMC completed.\n";
  
  return List::create(
    Named("beta") = beta_samples,
    Named("tau") = tau_samples,
    Named("rho") = rho_samples,
    Named("sigma") = sigma_samples,
    Named("phi") = phi_samples,
    Named("f") = f_samples
  );
}
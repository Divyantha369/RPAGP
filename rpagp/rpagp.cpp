// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <omp.h>
#include <map>
#include <tuple>
#include <string>
#include <vector>
#include <limits>
using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

//  Caching Structures 

// Cache key structures
struct KernelKey {
    int n;
    double rho;
    double alpha;
    double nugget;
    
    bool operator<(const KernelKey& other) const {
        if (n != other.n) return n < other.n;
        if (rho != other.rho) return rho < other.rho;
        if (alpha != other.alpha) return alpha < other.alpha;
        return nugget < other.nugget;
    }
};

struct KiKey {
    int n;
    double rho;
    double tau;
    double beta;
    
    bool operator<(const KiKey& other) const {
        if (n != other.n) return n < other.n;
        if (rho != other.rho) return rho < other.rho;
        if (tau != other.tau) return tau < other.tau;
        return beta < other.beta;
    }
};

struct SigmaNuKey {
    int n;
    double sigma;
    std::string phi_str;
    
    bool operator<(const SigmaNuKey& other) const {
        if (n != other.n) return n < other.n;
        if (sigma != other.sigma) return sigma < other.sigma;
        return phi_str < other.phi_str;
    }
};

// Global caches 
std::map<KernelKey, arma::mat> kernel_cache;
std::map<KiKey, arma::mat> k_i_cache;
std::map<SigmaNuKey, arma::mat> sigma_nu_cache;

// Cache size limits to prevent memory bloat
const size_t MAX_KERNEL_CACHE_SIZE = 30;
const size_t MAX_KI_CACHE_SIZE = 200;
const size_t MAX_SIGMA_NU_CACHE_SIZE = 30;

// Helper function to convert phi vector to string for caching
std::string phi_to_string(const arma::vec& phi) {
    std::ostringstream ss;
    for (size_t i = 0; i < phi.n_elem; i++) {
        ss << phi(i);
        if (i < phi.n_elem - 1) ss << ",";
    }
    return ss.str();
}

// Clear all caches
void clear_caches() {
    kernel_cache.clear();
    k_i_cache.clear();
    sigma_nu_cache.clear();
}

//  Kernel Functions 

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
            const double dist = x[i] - x[j];
            const double dist_sq = dist * dist;
            K(i, j) = alpha_sq * std::exp(-rho_sq_half * dist_sq);
            K(j, i) = K(i, j);  // Symmetric matrix
        }
    }
    
    return K;
}

// Cached version of sq_exp_kernel_cpp
arma::mat cached_sq_exp_kernel_cpp(const arma::vec& x, double rho, double alpha = 1.0, double nugget = 0.0) {
    const int n = x.n_elem;
    
    // Create cache key
    KernelKey key = {n, rho, alpha, nugget};
    
    // Check if in cache
    auto it = kernel_cache.find(key);
    if (it != kernel_cache.end()) {
        return it->second;
    }
    
    // Not in cache, compute it
    arma::mat K = sq_exp_kernel_cpp(x, rho, alpha, nugget);
    
    // Add to cache if not too large
    if (kernel_cache.size() < MAX_KERNEL_CACHE_SIZE) {
        kernel_cache[key] = K;
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
    
    // Precompute shifted time points
    arma::vec x_shifted = x - tau;
    
    for(int i = 0; i < n_time; i++) {
        const double x_i = x[i];
        for(int j = 0; j < n_time; j++) {
            // Note: This kernel is NOT symmetric when tau != 0
            const double diff = x_i - x_shifted[j];
            K(i, j) = beta * std::exp(-rho_sq_half * diff * diff);
        }
    }
    
    return K;
}

// Cached version of get_K_i_cpp
arma::mat cached_get_K_i_cpp(const arma::vec& x, double rho, double tau, double beta) {
    const int n = x.n_elem;
    
    // Create cache key
    KiKey key = {n, rho, tau, beta};
    
    // Check if in cache
    auto it = k_i_cache.find(key);
    if (it != k_i_cache.end()) {
        return it->second;
    }
    
    // Not in cache, compute it
    arma::mat K = get_K_i_cpp(x, rho, tau, beta);
    
    // Add to cache if not too large
    if (k_i_cache.size() < MAX_KI_CACHE_SIZE) {
        k_i_cache[key] = K;
    }
    
    return K;
}

//  Matrix Utility Functions 

// Convert upper triangular to symmetric matrix
// @param m Upper triangular matrix
// [[Rcpp::export]]
arma::mat ultosymmetric_cpp(const arma::mat& m) {
    return m + m.t() - diagmat(m.diag());
}

// Prediction Functions 

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
    const int n_time = x.n_elem;
    const int n = tau.n_elem;
    arma::mat y_hat(n_time, n);
    
    const arma::vec f_transformed = K_f_inv * f;
    
    // Parallelize over trials 
    #pragma omp parallel for if(n > 8)
    for(int i = 0; i < n; i++) {
        // Use cached K_i matrix
        arma::mat K_i = cached_get_K_i_cpp(x, rho, tau[i], beta[i]);
        y_hat.col(i) = K_i * f_transformed;
    }
    
    return y_hat;
}

//  Statistical Functions 

// Calculate AR process covariance matrix
// @param phi AR coefficients
// @param sigma Error standard deviation
// @param n_time Number of time points
// [[Rcpp::export]]
arma::mat get_Sigma_nu_cpp(const arma::vec& phi, double sigma, int n_time) {
    // Create a AR(p) covariance structure
    const int p = phi.n_elem;
    arma::mat Sigma(n_time, n_time, arma::fill::zeros);
    
    // Compute autocovariances 
    arma::vec acf = arma::vec(n_time, arma::fill::zeros);
    const double sigma_sq = sigma * sigma;
    const double phi_squared_sum = arma::sum(phi % phi);
    
    // Avoid division by zero or near-zero
    if (phi_squared_sum >= 1.0) {
        acf(0) = sigma_sq * 100.0; // Large variance for nearly non-stationary processes
    } else {
        acf(0) = sigma_sq / (1.0 - phi_squared_sum);
    }
    
    // AR(p) autocovariance calculation
    for(int k = 1; k < n_time; k++) {
        double sum = 0.0;
        for(int j = 0; j < std::min(p, k); j++) {
            sum += phi(j) * acf(k - j - 1);
        }
        acf(k) = sum;
    }
    
    // Fill the covariance matrix with the autocovariances
    for(int i = 0; i < n_time; i++) {
        Sigma(i, i) = acf(0); // Diagonal
        for(int j = 0; j < i; j++) {
            Sigma(i, j) = acf(i - j);
            Sigma(j, i) = Sigma(i, j); // Symmetric
        }
    }
    
    return Sigma;
}

// Cached version of get_Sigma_nu_cpp
arma::mat cached_get_Sigma_nu_cpp(const arma::vec& phi, double sigma, int n_time) {
    std::string phi_str = phi_to_string(phi);
    
    // Create cache key
    SigmaNuKey key = {n_time, sigma, phi_str};
    
    // Check if in cache
    auto it = sigma_nu_cache.find(key);
    if (it != sigma_nu_cache.end()) {
        return it->second;
    }
    
    // Not in cache, compute it
    arma::mat Sigma = get_Sigma_nu_cpp(phi, sigma, n_time);
    
    // Add to cache if not too large
    if (sigma_nu_cache.size() < MAX_SIGMA_NU_CACHE_SIZE) {
        sigma_nu_cache[key] = Sigma;
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
    
    //  symmetry
    Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0;
    return Sigma_i;
}

// Multivariate normal density calculation 
// @param x Vector of observations
// @param mean Mean vector
// @param sigma Covariance matrix
// [[Rcpp::export]]
double dmvnorm_cpp(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma) {
    const int n = x.n_elem;
    double log_det_val;
    double sign;
    
    // Try Cholesky decomposition for log determinant 
    arma::mat chol_sigma;
    bool chol_success = arma::chol(chol_sigma, sigma);
    
    if (chol_success) {
        // If Cholesky decomposition succeeded, use it for log determinant
        log_det_val = 2.0 * arma::sum(arma::log(arma::diagvec(chol_sigma)));
        sign = 1.0;
        
        // Solve linear system using Cholesky 
        arma::vec x_centered = x - mean;
        arma::vec z = arma::solve(arma::trimatl(chol_sigma.t()), x_centered);
        double quadform = arma::dot(z, z);
        
        return -0.5 * (n * log(2.0 * M_PI) + log_det_val + quadform);
    } else {
        // Fall back to standard log determinant
        arma::log_det(log_det_val, sign, sigma);
        
        if (sign <= 0) {
            return -1e10; // Return large negative number 
        }
        
        arma::vec x_centered = x - mean;
        
        // Use solve instead of explicit inverse when possible
        double quadform;
        bool solve_success = arma::solve(quadform, sigma, x_centered, x_centered);
        
        if (!solve_success) {
            // Fall back to explicit inverse as last resort
            quadform = arma::as_scalar(x_centered.t() * arma::inv(sigma) * x_centered);
        }
        
        return -0.5 * (n * log(2.0 * M_PI) + log_det_val + quadform);
    }
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
    const int n = y.n_cols;
    const int n_time = y.n_rows;
    
    // Create Sigma_nu (AR covariance) using cached version
    const arma::mat Sigma_nu = cached_get_Sigma_nu_cpp(phi, sigma, n_time);
    const arma::vec f_transformed = K_f_inv * f;
    
    double tmp = 0.0;
    
    // Precompute matrices for each trial to avoid redundant computations
    std::vector<arma::mat> K_i_storage(n);
    std::vector<arma::mat> Sigma_storage(n);
    std::vector<arma::vec> mu_storage(n);
    
    // Prepare computations first without parallelization
    for(int i = 0; i < n; i++) {
        // Use cached K_i
        K_i_storage[i] = cached_get_K_i_cpp(x, rho, tau[i], beta[i]);
        
        // Compute Sigma_y_i_f
        Sigma_storage[i] = getSigma_y_i_f_cpp(beta[i], K_f, K_f_inv, K_i_storage[i], Sigma_nu);
        
        // Ensure symmetry
        Sigma_storage[i] = (Sigma_storage[i] + Sigma_storage[i].t()) / 2.0;
        
        // Compute means using pre-computed f_transformed
        mu_storage[i] = K_i_storage[i] * f_transformed;
    }
    
    // Parallelize the likelihood computations
    #pragma omp parallel for reduction(+:tmp) if(n > 8)
    for(int i = 0; i < n; i++) {
    
        tmp += dmvnorm_cpp(y.col(i), mu_storage[i], Sigma_storage[i]);
    }
    
    // Handle extreme negative values
    if (tmp < -1e9) {
        return -1e10;
    }
    
    return tmp;
}

//  MCMC Sampling Functions

// Tau proposal for Metropolis-Hastings
// @param tau Current tau values
// @param tau_proposal_sd Proposal standard deviation
// [[Rcpp::export]]
arma::vec propose_tau_cpp(const arma::vec& tau, double tau_proposal_sd) {
    const int n = tau.n_elem;
    const arma::mat Sigma = tau_proposal_sd * tau_proposal_sd * arma::eye(n, n);
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
    const int n = tau.n_elem;
    const arma::mat Sigma = tau_prior_sd * tau_prior_sd * (arma::eye(n, n) - arma::ones(n, n) / (n + 1.0));
    
    const arma::vec mean = arma::zeros(n);
    double log_det_val;
    double sign;
    arma::log_det(log_det_val, sign, Sigma);
    
    if (sign <= 0) {
        return -1e10;
    }
    
    double quadform;
    bool solve_success = arma::solve(quadform, Sigma, tau, tau);
    
    if (!solve_success) {
        // Fall back to inverse if necessary
        quadform = arma::as_scalar(tau.t() * arma::inv(Sigma) * tau);
    }
    
    const double logdens = -0.5 * (n * log(2.0 * M_PI) + log_det_val + quadform);
    
    return logdens;
}

// Rho prior evaluation (gamma distribution)
// @param rho Rho value
// @param shape Shape parameter
// @param scale Scale parameter
// [[Rcpp::export]]
double prior_rho_cpp(double rho, double shape, double scale) {
    if (rho <= 0) return -1e10;
    
    const double logdens = (shape - 1.0) * log(rho) - rho / scale - shape * log(scale) - lgamma(shape);
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
    // Create kernel matrices with caching
    const arma::mat K_f = cached_sq_exp_kernel_cpp(x, current_rho, 1.0, 1e-6);
    const arma::mat K_f_inv = arma::inv_sympd(K_f);
    
    // Propose new tau
    const arma::vec proposed_tau = propose_tau_cpp(current_tau, tau_proposal_sd);
    
    // Compute likelihoods and priors
    const double lik_current = likelihood_cpp(y, f, current_tau, current_beta, current_rho, 
                                    current_sigma, current_phi, x, K_f, K_f_inv);
    const double prior_current = prior_tau_cpp(current_tau, tau_prior_sd);
    
    const double lik_proposed = likelihood_cpp(y, f, proposed_tau, current_beta, current_rho, 
                                     current_sigma, current_phi, x, K_f, K_f_inv);
    const double prior_proposed = prior_tau_cpp(proposed_tau, tau_prior_sd);
    
    // Metropolis-Hastings step
    const double log_ratio = lik_proposed + prior_proposed - lik_current - prior_current;
    const double acceptance_prob = std::min(1.0, exp(log_ratio));
    
    // Accept or reject
    const double u = arma::as_scalar(arma::randu(1));
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
    const double proposed_rho = propose_rho_cpp(current_rho, rho_proposal_sd);
    
    // Skip computation if proposed rho is negative
    if (proposed_rho <= 0) {
        return current_rho;
    }
    
    // Compute kernel matrices for current and proposed with caching
    const arma::mat K_f_curr = cached_sq_exp_kernel_cpp(x, current_rho, 1.0, 1e-6);
    const arma::mat K_f_curr_inv = arma::inv_sympd(K_f_curr);
    
    const arma::mat K_f_prop = cached_sq_exp_kernel_cpp(x, proposed_rho, 1.0, 1e-6);
    const arma::mat K_f_prop_inv = arma::inv_sympd(K_f_prop);
    
    // Compute likelihoods and priors
    const double lik_current = likelihood_cpp(y, f, current_tau, current_beta, current_rho, 
                                    current_sigma, current_phi, x, K_f_curr, K_f_curr_inv);
    const double prior_current = prior_rho_cpp(current_rho, rho_prior_shape, rho_prior_scale);
    
    const double lik_proposed = likelihood_cpp(y, f, current_tau, current_beta, proposed_rho, 
                                     current_sigma, current_phi, x, K_f_prop, K_f_prop_inv);
    const double prior_proposed = prior_rho_cpp(proposed_rho, rho_prior_shape, rho_prior_scale);
    
    // Metropolis-Hastings step
    const double log_ratio = lik_proposed + prior_proposed - lik_current - prior_current;
    const double acceptance_prob = std::min(1.0, exp(log_ratio));
    
    // Accept or reject
    const double u = arma::as_scalar(arma::randu(1));
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
    const int n = y_i.n_elem;
    
    // Modified design matrix (scaled by current beta)
    const arma::vec X = y_hat_i / current_beta_i;
    
    // Posterior calculations for Bayesian linear regression
    const double X_sq_sum = arma::dot(X, X);
    const double prior_variance = beta_prior_sd * beta_prior_sd;
    const double V_post = 1.0 / (1.0 / prior_variance + X_sq_sum);
    const double mu_post = V_post * (beta_prior_mu / prior_variance + arma::dot(X, y_i));
    
    const double a_post = 1.0 + n / 2.0;
    const double b_post = 1.0 + 0.5 * (beta_prior_mu * beta_prior_mu / prior_variance + 
                                arma::dot(y_i, y_i) - mu_post * mu_post / V_post);
    
    // Sample from scaled inverse chi-squared (via inverse gamma)
    const double sigma2_sample = 1.0 / arma::randg(arma::distr_param(a_post, 1.0 / b_post));
    
    // Sample beta from normal
    const double beta_sample = arma::as_scalar(arma::randn(1)) * sqrt(V_post * sigma2_sample) + mu_post;
    
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
    const int n = y.n_cols;
    arma::vec new_beta = current_beta;
    
    // Parallelize beta sampling across trials
    #pragma omp parallel for if(n > 8)
    for (int i = 0; i < n; i++) {
        new_beta(i) = sample_beta_single_cpp(y.col(i), y_hat.col(i), current_beta(i), 
                                          beta_prior_mu, beta_prior_sd);
    }
    
    return new_beta;
}

// Sample f using reduction pattern
// [[Rcpp::export]]
arma::mat sample_f_parallel_cpp(const arma::mat& y, const arma::vec& tau, const arma::vec& beta,
                              double rho, double sigma, const arma::vec& phi, 
                              const arma::vec& x, int n_draws, double nugget = 1e-6) {
    const int n_time = y.n_rows;
    const int n = y.n_cols;
    
    // Create K_f and K_f_inv with caching
    const arma::mat K_f = cached_sq_exp_kernel_cpp(x, rho, 1.0, nugget);
    const arma::mat K_f_inv = arma::inv_sympd(K_f);
    
    // Create Sigma_nu with caching
    const arma::mat Sigma_nu = cached_get_Sigma_nu_cpp(phi, sigma, n_time);
    
    // Create output matrix for f draws
    arma::mat f_draws(n_time, n_draws);
    
    // Pre-allocate matrices 
    arma::mat A(n_time, n_time);
    arma::vec b(n_time);
    
    std::vector<arma::mat> K_i_matrices(n);
    std::vector<arma::mat> Sigma_i_matrices(n);
    std::vector<arma::mat> Sigma_i_inv_matrices(n);
    
    // Prepare matrices that can be reused across draws
    for (int i = 0; i < n; i++) {
        K_i_matrices[i] = cached_get_K_i_cpp(x, rho, tau[i], beta[i]);
        const arma::mat Sigma_y_i = get_Sigma_y_i_cpp(beta[i], K_f, Sigma_nu);
        Sigma_i_matrices[i] = Sigma_y_i - K_i_matrices[i].t() * K_f_inv * K_i_matrices[i];
        Sigma_i_matrices[i] = (Sigma_i_matrices[i] + Sigma_i_matrices[i].t()) / 2.0; // Ensure symmetry
        
        // Compute inverse or pseudo-inverse
        bool inv_success = arma::inv_sympd(Sigma_i_inv_matrices[i], Sigma_i_matrices[i]);
        if (!inv_success) {
            // Try regular inverse if sympd fails
            bool reg_inv_success = arma::inv(Sigma_i_inv_matrices[i], Sigma_i_matrices[i]);
            if (!reg_inv_success) {
                // Fall back to pseudo-inverse as last resort
                Sigma_i_inv_matrices[i] = arma::pinv(Sigma_i_matrices[i]);
            }
        }
    }
    
    for (int iter = 0; iter < n_draws; iter++) {
        A = K_f_inv;
        b.zeros();
        
        // Process each trial to build A and b
        for (int i = 0; i < n; i++) {
            const arma::mat& K_i = K_i_matrices[i];
            const arma::mat& Sigma_i_inv = Sigma_i_inv_matrices[i];
            
            const arma::mat L = K_i * K_f_inv;
            const arma::mat G = Sigma_i_inv * L;
            
            A += L.t() * G;
            b += L.t() * Sigma_i_inv * y.col(i);
        }
        
        // Ensure A is symmetric 
        A = (A + A.t()) / 2.0;
        
        // Compute posterior covariance and mean
        arma::mat K_f_post;
        bool inv_success = arma::inv_sympd(K_f_post, A);
        if (!inv_success) {
            // Try regular inverse if sympd fails
            bool reg_inv_success = arma::inv(K_f_post, A);
            if (!reg_inv_success) {
                // Fall back to pseudo-inverse as last resort
                K_f_post = arma::pinv(A);
            }
        }
        
        // Ensure symmetry
        K_f_post = (K_f_post + K_f_post.t()) / 2.0;
        
        // Compute posterior mean
        const arma::vec mean = K_f_post * b;
        
        // Sample from multivariate normal
        arma::mat L;
        bool chol_success = arma::chol(L, K_f_post, "lower");
        
        if (chol_success) {
            const arma::vec z = arma::randn(n_time);
            f_draws.col(iter) = mean + L * z;
        } else {
            // Fallback if Cholesky decomposition fails
            arma::mat U;
            arma::vec d;
            arma::eig_sym(d, U, K_f_post);
            
            // Ensure all eigenvalues are positive
            d = arma::max(d, arma::zeros(d.n_elem));
            
            // Construct sample
            const arma::vec z = arma::randn(n_time);
            f_draws.col(iter) = mean + U * arma::diagmat(arma::sqrt(d)) * z;
        }
    }
    
    return f_draws;
}

// AR simulation for residuals 
// @param n Length of series to generate
// @param phi AR parameters
// @param sigma Innovation standard deviation
// [[Rcpp::export]]
arma::vec arima_sim_cpp(int n, const arma::vec& phi, double sigma) {
    const int p = phi.n_elem;
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
    const int n_time = z.n_rows;
    const int n = z.n_cols;
    
    // Flatten the residuals 
    arma::vec z_vec(n_time * n);
    for (int i = 0; i < n; i++) {
        z_vec.subvec(i * n_time, (i + 1) * n_time - 1) = z.col(i);
    }
    
    // Set up the design matrix X for the AR model 
    const int total_rows = n_time * n - ar_order * n;
    arma::mat X(total_rows, ar_order);
    arma::vec y(total_rows);
    
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
    
    // Sample phi - use QR decomposition 
    arma::mat Q, R;
    bool qr_success = arma::qr(Q, R, X);
    
    arma::vec phi_mean;
    if (qr_success) {
        phi_mean = arma::solve(R, Q.t() * y);
    } else {
        // Fall back to normal equations with regularization
        const arma::mat XtX = X.t() * X;
        const arma::mat XtX_inv = arma::inv(XtX + arma::eye(ar_order, ar_order) * 1e-6);
        phi_mean = XtX_inv * X.t() * y;
    }
    
    // Check stationarity
    const arma::cx_vec roots = arma::roots(arma::join_cols(arma::ones(1), -phi_mean));
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
    const arma::vec eps = y - X * phi_mean;
    const double sigma2 = arma::dot(eps, eps) / (y.n_elem - ar_order);
    
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
    const arma::mat K_f = cached_sq_exp_kernel_cpp(x, rho, 1.0, 1e-6);
    const arma::mat K_f_inv = arma::inv_sympd(K_f);
    
    // Cholesky decomposition for sampling
    arma::mat L;
    bool success = arma::chol(L, K_f, "lower");
    
    // Generate f
    arma::vec f;
    if (success) {
        const arma::vec z = arma::randn(n_time);
        f = L * z;
    } else {
        // Fallback using eigendecomposition
        arma::vec eigval;
        arma::mat eigvec;
        arma::eig_sym(eigval, eigvec, K_f);
        
        // Ensure positive eigenvalues
        eigval = arma::max(eigval, arma::zeros(eigval.n_elem));
        
        // Generate f
        const arma::vec z = arma::randn(n_time);
        f = eigvec * arma::diagmat(arma::sqrt(eigval)) * z;
    }
    
    // Initialize matrices
    arma::mat y(n_time, n);
    arma::mat z(n_time, n);
    arma::mat mu(n_time, n);
    
    // Pre-compute K_f_inv * f
    const arma::vec f_transformed = K_f_inv * f;
    
    // Generate trial data in parallel
    #pragma omp parallel for if(n > 8)
    for (int i = 0; i < n; i++) {
        // Compute K_i
        const arma::mat K_i = cached_get_K_i_cpp(x, rho, tau(i), beta(i));
        
        // Compute mean using pre-computed f_transformed
        mu.col(i) = K_i * f_transformed;
        
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
    // Clear caches at the beginning of MCMC
    clear_caches();
    
    // Get dimensions
    const int n_time = y.n_rows;
    const int n = y.n_cols;
    const int p = initial_phi.n_elem;
    
    // Set number of OpenMP threads
    #ifdef _OPENMP
    int num_cores = omp_get_num_procs();
    omp_set_num_threads(std::min(num_cores, 8)); // Limit to 8 threads maximum
    #endif
    
    // Create vector of time points
    const arma::vec x = arma::linspace(0, 1, n_time);
    
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
    
    // Cache for latest parameters
    arma::mat K_f = cached_sq_exp_kernel_cpp(x, initial_rho, 1.0, 1e-6);
    arma::mat K_f_inv = arma::inv_sympd(K_f);
    double cached_rho = initial_rho;
    
    // Cache for Sigma_nu
    arma::mat Sigma_nu = cached_get_Sigma_nu_cpp(initial_phi, initial_sigma, n_time);
    double cached_sigma = initial_sigma;
    arma::vec cached_phi = initial_phi;
    
    // Initial f sample
    arma::mat f_draw = sample_f_parallel_cpp(y, initial_tau, initial_beta, initial_rho, 
                                           initial_sigma, initial_phi, x, 1, 1e-6);
    f_samples.col(0) = f_draw.col(0);
    
    // Ensure pinned value
    f_samples.col(0) = pinned_value * f_samples.col(0) / f_samples(pinned_point, 0);
    
    // Pre-allocate y_hat matrix t
    arma::mat y_hat(n_time, n);
    
    // MCMC loop
    for (int iter = 1; iter < n_iter; iter++) {
        if (iter % (n_iter/10) == 0) {
            Rcpp::Rcout << "... " << static_cast<int>((iter * 100.0) / n_iter) << "% \n";
            
            // Periodically clean caches to prevent memory bloat
            if (kernel_cache.size() > MAX_KERNEL_CACHE_SIZE / 2 ||
                k_i_cache.size() > MAX_KI_CACHE_SIZE / 2 ||
                sigma_nu_cache.size() > MAX_SIGMA_NU_CACHE_SIZE / 2) {
                clear_caches();
            }
        }
        
        // Get current parameters
        arma::vec current_beta = beta_samples.col(iter-1);
        arma::vec current_tau = tau_samples.col(iter-1);
        double current_rho = rho_samples(iter-1);
        double current_sigma = sigma_samples(iter-1);
        arma::vec current_phi = phi_samples.col(iter-1);
        arma::vec current_f = f_samples.col(iter-1);
        
        // Update K_f and K_f_inv only if rho changed
        if (current_rho != cached_rho) {
            K_f = cached_sq_exp_kernel_cpp(x, current_rho, 1.0, 1e-6);
            K_f_inv = arma::inv_sympd(K_f);
            cached_rho = current_rho;
        }
        
        // Update Sigma_nu only if sigma or phi changed
        if (current_sigma != cached_sigma || arma::any(current_phi != cached_phi)) {
            Sigma_nu = cached_get_Sigma_nu_cpp(current_phi, current_sigma, n_time);
            cached_sigma = current_sigma;
            cached_phi = current_phi;
        }
        
        // Sample f and rescale
        f_draw = sample_f_parallel_cpp(y, current_tau, current_beta, current_rho, 
                                      current_sigma, current_phi, x, 1, 1e-6);
        current_f = f_draw.col(0);
        current_f = pinned_value * current_f / current_f(pinned_point);
        
        // Get y_hat -
        y_hat = get_y_hat_matrix_cpp(current_f, x, current_rho, current_tau, current_beta, K_f_inv);
        
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
        
        // Update K_f and K_f_inv if rho changed
        if (current_rho != cached_rho) {
            K_f = cached_sq_exp_kernel_cpp(x, current_rho, 1.0, 1e-6);
            K_f_inv = arma::inv_sympd(K_f);
            cached_rho = current_rho;
        }
        
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
    
    // Clear caches at the end of MCMC
    clear_caches();
    
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

// Clear caches and reset function
// [[Rcpp::export]]
void reset_caches_cpp() {
    clear_caches();
    Rcpp::Rcout << "All computation caches have been cleared.\n";
}

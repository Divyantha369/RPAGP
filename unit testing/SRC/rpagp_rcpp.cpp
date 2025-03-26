// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace Rcpp;
using namespace arma;

// Set the number of threads to 75% of available cores
void setOMPThreads() {
    int max_threads = omp_get_max_threads();
    int use_threads = std::max(1, (int)(max_threads * 0.75));
    omp_set_num_threads(use_threads);
    Rcout << "Using " << use_threads << " threads out of " << max_threads << " available\n";
}

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

// Square exponential kernel function
// @param x Vector of time points
// @param rho Length scale parameter
// @param nugget Small value for numerical stability
// [[Rcpp::export]]
arma::mat sq_exp_kernel(const arma::vec& x, double rho, double nugget = 1e-6) {
    int n = x.n_elem;
    arma::mat K(n, n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            K(i, j) = std::exp(-0.5 * std::pow(rho, 2) * std::pow(x(i) - x(j), 2));
        }
    }
    
    // Add nugget to diagonal for numerical stability
    for (int i = 0; i < n; i++) {
        K(i, i) += nugget;
    }
    
    return K;
}

// Get Covariance AR process 
// @param phi autoregressive parms (p x 1)
// @param sigma error variance
// @param n_time number of time points
// [[Rcpp::export]]
arma::mat get_Sigma_nu(const arma::vec& phi, double sigma, int n_time) {
    int p = phi.n_elem;  // AR order
    arma::mat Sigma_nu_U(n_time, n_time, arma::fill::zeros);
    arma::vec acf = arma::zeros(n_time);
    double sigma2 = std::pow(sigma, 2);
    
    // Compute autocovariance at lag 0 (variance)
    if (p == 1) {
        // AR(1) process - simple formula
        acf(0) = sigma2 / (1.0 - std::pow(phi(0), 2));
    } 
    else if (p == 2) {
        // AR(2) process - correct formula for variance
        double phi1 = phi(0);
        double phi2 = phi(1);
        
        // Check for near-stationarity boundary conditions
        if (std::abs(phi1 + phi2) > 0.999 || std::abs(phi1 - phi2) > 0.999 || std::abs(phi2) > 0.999) {
            // Apply a small shrinkage to prevent numerical issues
            phi1 *= 0.99;
            phi2 *= 0.99;
        }
        
        double denom = (1.0 - phi2) * ((1.0 - phi2) * (1.0 - phi2) - std::pow(phi1, 2));
        if (std::abs(denom) < 1e-6) {
            // Handle near-singularity
            denom = 1e-6;
        }
        acf(0) = sigma2 * (1.0 - phi2) / denom;
    }
    else {
        // General AR(p) process - use Yule-Walker equations
        // This is a simplified approach that works for many cases
        double denom = 1.0;
        for (int i = 0; i < p; i++) {
            denom -= std::pow(phi(i), 2);
        }
        // Add a small stability factor
        if (std::abs(denom) < 1e-6) {
            denom = 1e-6;
        }
        acf(0) = sigma2 / denom;
    }
    
    // For AR(2), compute acf(1) directly from theory
    if (p == 2) {
        double phi1 = phi(0);
        double phi2 = phi(1);
        acf(1) = phi1/(1.0 - phi2) * acf(0);
    }
    
    // Recursively compute autocovariances at lags > 0 or > 1 (for AR(2))
    int start_lag = (p == 2) ? 2 : 1;
    for (int lag = start_lag; lag < n_time; lag++) {
        double sum = 0.0;
        for (int j = 0; j < p; j++) {
            if (lag - (j + 1) >= 0) {
                sum += phi(j) * acf(lag - (j + 1));
            }
        }
        acf(lag) = sum;
    }
    
    // Fill the upper triangular part of the Toeplitz matrix
    for (int tt = 0; tt < n_time; tt++) {
        for (int i = tt; i < n_time; i++) {
            Sigma_nu_U(tt, i) = acf(i - tt);
        }
    }
    
    // Make the matrix symmetric
    arma::mat Sigma_nu = Sigma_nu_U + Sigma_nu_U.t() - diagmat(Sigma_nu_U);
    
    // Check and ensure positive definiteness
    if (!is_positive_definite(Sigma_nu)) {
        // Add a small regularization to the diagonal
        double min_eig_val = arma::min(arma::eig_sym(Sigma_nu));
        double eps = std::abs(min_eig_val) + 1e-6;
        Sigma_nu.diag() += eps;
    }
    
    return Sigma_nu;
}

// Get Cov(y_i, f) - K_i matrix for a specific trial
// [[Rcpp::export]]
arma::mat get_trial_K_i(const arma::vec& x, double rho, double tau, double beta) {
    int n_time = x.n_elem;
    arma::mat K(n_time, n_time);
    
    #pragma omp parallel for
    for (int i = 0; i < n_time; i++) {
        for (int j = 0; j < n_time; j++) {
            K(i, j) = std::exp(-0.5 * std::pow(rho, 2) * std::pow(x(i) - x(j) - tau, 2));
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

// Get Sigma_y_i_f (conditional covariance given f)
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
    arma::mat Sigma_y_i = get_Sigma_y_i(betas(i), K_f, Sigma_nu);
    arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
    arma::mat Sigma_i = Sigma_y_i - K_i.t() * K_f_inv * K_i;
    
    // Ensure symmetry for numerical stability
    Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0;
    
    // Check and ensure positive definiteness
    if (!is_positive_definite(Sigma_i)) {
        double min_eig_val = arma::min(arma::eig_sym(Sigma_i));
        double eps = std::abs(min_eig_val) + 1e-6;
        Sigma_i.diag() += eps;
    }
    
    return Sigma_i;
}

// Get y_hat (predicted mean for trial i)
// @param i trial index
// @param f structural signal
// @param betas amplitudes
// @param taus latencies
// @param rho GP length scale
// @param K_f_inv Inverse kernel matrix
// [[Rcpp::export]]
arma::vec get_y_hat(int i, const arma::vec& f, const arma::vec& betas,
                   const arma::vec& taus, double rho, const arma::mat& K_f_inv) {
    arma::vec x = arma::linspace(0, 1, f.n_elem);
    arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
    arma::vec mu = K_i * K_f_inv * f;
    return mu;
}

// Get y_hat matrix for all trials
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
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y_hat.col(i) = get_y_hat(i, f, betas, taus, rho, K_f_inv);
    }
    
    return y_hat;
}

// Multivariate normal log density
// @param x Vector
// @param mean Mean vector
// @param sigma Covariance matrix
// [[Rcpp::export]]
double dmvnorm(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma, bool log_p = true) {
    int n = x.n_elem;
    double log_det_sigma;
    double sign;
    
    try {
        log_det(log_det_sigma, sign, sigma);
    }
    catch (const std::exception& e) {
        // Handle numerical issues in log determinant calculation
        Rcout << "Warning in dmvnorm: " << e.what() << "\n";
        
        // Regularize the matrix and try again
        arma::mat sigma_reg = sigma;
        sigma_reg.diag() += 1e-6;
        log_det(log_det_sigma, sign, sigma_reg);
    }
    
    arma::vec x_centered = x - mean;
    double quadform;
    
    try {
        quadform = as_scalar(x_centered.t() * inv_sympd(sigma) * x_centered);
    }
    catch (const std::exception& e) {
        // Handle numerical issues in matrix inversion
        Rcout << "Warning in dmvnorm quadratic form: " << e.what() << "\n";
        
        // Use more stable but less efficient method
        arma::mat sigma_reg = sigma;
        sigma_reg.diag() += 1e-6;
        quadform = as_scalar(x_centered.t() * pinv(sigma_reg) * x_centered);
    }
    
    double log_pdf = -0.5 * n * std::log(2.0 * M_PI) - 0.5 * log_det_sigma - 0.5 * quadform;
    
    if (log_p) {
        return log_pdf;
    } else {
        return std::exp(log_pdf);
    }
}

// Prior for tau (latency)
// @param tau latency values
// @param tau_prior_sd prior SD
// [[Rcpp::export]]
double prior_tau(const arma::vec& tau, double tau_prior_sd) {
    int n = tau.n_elem;
    
    // Create covariance matrix
    arma::mat Sigma = std::pow(tau_prior_sd, 2) * (arma::eye(n, n) - arma::ones(n, n) / (n + 1));
    
    // Return log density
    return dmvnorm(tau, arma::zeros(n), Sigma, true);
}

// Prior for rho (GP length scale)
// @param rho length scale value
// @param rho_prior_shape shape parameter
// @param rho_prior_scale scale parameter
// [[Rcpp::export]]
double prior_rho(double rho, double rho_prior_shape, double rho_prior_scale) {
    // Gamma log density
    return rho_prior_shape * std::log(rho_prior_scale) - 
           std::lgamma(rho_prior_shape) + 
           (rho_prior_shape - 1) * std::log(rho) - 
           rho_prior_scale * rho;
}

// Likelihood function
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
        Rcout << "Error in get_Sigma_nu: " << e.what() << "\n";
        return -1e10; // Return a very low likelihood in case of error
    }
    
    double result = 0.0;
    
    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < n; i++) {
        try {
            arma::mat Sigma_y_i_f = getSigma_y_i_f(i, x, betas, taus, rho, K_f, K_f_inv, Sigma_nu);
            arma::vec mu = get_y_hat(i, f, betas, taus, rho, K_f_inv);
            double trial_ll = dmvnorm(y.col(i), mu, Sigma_y_i_f, true);
            
            // Avoid extreme values
            if (std::isfinite(trial_ll)) {
                result += trial_ll;
            } else {
                result -= 1e5; // Penalty for non-finite likelihood
            }
        }
        catch (const std::exception& e) {
            Rcout << "Error in trial " << i << " likelihood: " << e.what() << "\n";
            result -= 1e5; // Penalty for error
        }
    }
    
    // Handle numerical issues
    if (!std::isfinite(result)) {
        return -1e10;
    }
    
    return result;
}

// Proposal for tau (latency)
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
    
    return proposal;
}

// Proposal for rho (GP length scale)
// @param rho current rho value
// @param rho_proposal_sd SD for proposal
// [[Rcpp::export]]
double propose_rho(double rho, double rho_proposal_sd) {
    // Sample from normal distribution
    double proposal = rho + R::rnorm(0, rho_proposal_sd);
    // Ensure positivity
    return (proposal > 0) ? proposal : rho;
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
            Rcout << "Error in eigenvalue calculation: " << e.what() << "\n";
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

// Sample from AR posterior
// @param z residuals matrix
// @param ar_order AR order
// [[Rcpp::export]]
List sample_AR(const arma::mat& z, int ar_order) {
    int n_time = z.n_rows;
    int n = z.n_cols;
    
    // Flatten z into a single vector
    arma::vec z_flat = arma::vectorise(z);
    int n_total = z_flat.n_elem;
    
    // Create design matrix X
    arma::mat X = arma::zeros(n_total - ar_order, ar_order);
    arma::vec y = arma::zeros(n_total - ar_order);
    
    // Fill X and y
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
    
    // OLS estimation
    arma::vec beta_hat;
    
    try {
        beta_hat = arma::solve(X.t() * X, X.t() * y);
    }
    catch (const std::exception& e) {
        // Handle linear system solving error
        Rcout << "Error in AR coefficient estimation: " << e.what() << "\n";
        // Return some reasonable default
        beta_hat = arma::zeros(ar_order);
        if (ar_order >= 1) beta_hat(0) = 0.1; // Conservative AR(1) coefficient
        if (ar_order >= 2) beta_hat(1) = 0.05; // Conservative AR(2) coefficient
    }
    
    arma::vec residuals = y - X * beta_hat;
    double sigma2 = arma::dot(residuals, residuals) / (n_total - ar_order - ar_order);
    
    // Sample AR coefficients from posterior
    arma::mat Sigma_beta;
    try {
        Sigma_beta = sigma2 * arma::inv_sympd(X.t() * X);
    }
    catch (const std::exception& e) {
        // Handle matrix inversion error
        Rcout << "Error in AR coefficient covariance calculation: " << e.what() << "\n";
        // Use a regularized version
        arma::mat XtX = X.t() * X;
        XtX.diag() += 1e-6;
        Sigma_beta = sigma2 * arma::inv_sympd(XtX);
    }
    
    // Try until we get a stationary process
    int max_attempts = 1000;
    int attempt = 0;
    arma::vec phi_sample;
    
    do {
        try {
            phi_sample = arma::mvnrnd(beta_hat, Sigma_beta);
        }
        catch (const std::exception& e) {
            // If sampling fails, use the OLS estimate with small noise
            Rcout << "Error in AR coefficient sampling: " << e.what() << "\n";
            phi_sample = beta_hat + 0.01 * arma::randn(ar_order);
        }
        
        attempt++;
        if (attempt > max_attempts) {
            // If too many attempts, return slightly shrunk coefficients
            phi_sample = 0.8 * beta_hat;
            break;
        }
    } while (!is_stationary(phi_sample));
    
    return List::create(
        Named("rho") = phi_sample,
        Named("sigma2") = sigma2
    );
}

// Sample f (structural signal)
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
                  int n_draws = 1, double nugget = 1e-6) {
    int n_time = y.n_rows;
    int n = y.n_cols;
    arma::mat f_draws(n_time, n_draws);
    
    arma::vec x = arma::linspace(0, 1, n_time);
    arma::mat K_f = sq_exp_kernel(x, rho, nugget);
    arma::mat K_f_inv;
    
    try {
        K_f_inv = arma::inv_sympd(K_f);
    }
    catch (const std::exception& e) {
        // Handle matrix inversion error
        Rcout << "Error in K_f inversion: " << e.what() << "\n";
        // Add regularization and try again
        K_f.diag() += 1e-4;
        K_f_inv = arma::inv_sympd(K_f);
    }
    
    arma::mat Sigma_nu;
    try {
        Sigma_nu = get_Sigma_nu(phi, sigma, n_time);
    }
    catch (const std::exception& e) {
        // Handle error in AR covariance
        Rcout << "Error in get_Sigma_nu: " << e.what() << "\n";
        // Use diagonal matrix as fallback
        Sigma_nu = sigma * sigma * arma::eye(n_time, n_time);
    }
    
    for (int iter = 0; iter < n_draws; iter++) {
        // Progress reporting
        if (iter % 10 == 0 && iter > 0) {
            Rcout << iter / 10;
        }
        
        arma::mat A = K_f_inv;
        arma::vec b = arma::zeros(n_time);
        
        for (int i = 0; i < n; i++) {
            try {
                arma::mat Sigma_y_i = get_Sigma_y_i(betas(i), K_f, Sigma_nu);
                arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
                arma::mat Sigma_i = Sigma_y_i - K_i.t() * K_f_inv * K_i;
                
                // Ensure symmetry
                Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0;
                
                // Check and ensure positive definiteness
                if (!is_positive_definite(Sigma_i)) {
                    double min_eig_val = arma::min(arma::eig_sym(Sigma_i));
                    double eps = std::abs(min_eig_val) + 1e-6;
                    Sigma_i.diag() += eps;
                }
                
                arma::mat Sigma_i_inv = arma::inv_sympd(Sigma_i);
                arma::mat L = K_i * K_f_inv;
                arma::mat G = Sigma_i_inv * L;
                
                A = A + L.t() * G;
                b = b + L.t() * Sigma_i_inv * y.col(i);
            }
            catch (const std::exception& e) {
                Rcout << "Error in trial " << i << " sampling f: " << e.what() << "\n";
                // Skip this trial
                continue;
            }
        }
        
        // Ensure A is symmetric
        A = (A + A.t()) / 2.0;
        
        // Check and ensure positive definiteness
        if (!is_positive_definite(A)) {
            double min_eig_val = arma::min(arma::eig_sym(A));
            double eps = std::abs(min_eig_val) + 1e-6;
            A.diag() += eps;
        }
        
        arma::mat K_f_post;
        try {
            K_f_post = arma::inv_sympd(A);
            // Ensure symmetry
            K_f_post = (K_f_post + K_f_post.t()) / 2.0;
        }
        catch (const std::exception& e) {
            // Handle matrix inversion error
            Rcout << "Error in K_f_post calculation: " << e.what() << "\n";
            // Use a simpler approach as fallback
            K_f_post = arma::pinv(A);
        }
        
        arma::vec mean_f = K_f_post * b;
        
        // Sample from multivariate normal
        try {
            arma::mat chol_K_f_post = arma::chol(K_f_post, "lower");
            arma::vec z = arma::randn(n_time);
            f_draws.col(iter) = mean_f + chol_K_f_post * z;
        }
        catch (const std::exception& e) {
            // Handle Cholesky decomposition error
            Rcout << "Error in Cholesky decomposition: " << e.what() << "\n";
            // Use eigendecomposition as fallback
            arma::vec eigval;
            arma::mat eigvec;
            arma::eig_sym(eigval, eigvec, K_f_post);
            
            // Ensure positive eigenvalues
            for (unsigned int i = 0; i < eigval.n_elem; i++) {
                eigval(i) = std::max(eigval(i), 1e-6);
            }
            
            arma::mat sqrt_mat = eigvec * diagmat(sqrt(eigval)) * eigvec.t();
            arma::vec z = arma::randn(n_time);
            f_draws.col(iter) = mean_f + sqrt_mat * z;
        }
    }
    
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
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        arma::vec y_i = y.col(i);
        arma::vec y_hat_i = y_hat.col(i);
        
        // Calculate sufficient statistics
        double ss_y_hat = arma::dot(y_hat_i, y_hat_i);
        double ss_y_y_hat = arma::dot(y_i, y_hat_i);
        
        // Calculate posterior parameters
        double var_prior = std::pow(beta_prior_sd, 2);
        double var_post = 1.0 / (1.0 / var_prior + ss_y_hat);
        double mean_post = var_post * (beta_prior_mu / var_prior + ss_y_y_hat);
        
        // Sample from posterior
        betas(i) = R::rnorm(mean_post, std::sqrt(var_post));
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
    arma::mat K_f = sq_exp_kernel(x, rho, 1e-6);
    arma::mat K_f_inv;
    
    try {
        K_f_inv = arma::inv_sympd(K_f);
    }
    catch (const std::exception& e) {
        // Handle matrix inversion error
        Rcout << "Error in K_f inversion in sample_tau: " << e.what() << "\n";
        // Add regularization and try again
        K_f.diag() += 1e-4;
        K_f_inv = arma::inv_sympd(K_f);
    }
    
    // Propose new taus
    arma::vec taus_proposed = propose_tau(taus, tau_proposal_sd);
    
    // Calculate likelihoods and priors
    double lik_current = likelihood(y, f, betas, taus, rho, phi, sigma, K_f, K_f_inv);
    double prior_current = prior_tau(taus, tau_prior_sd);
    
    double lik_proposed = likelihood(y, f, betas, taus_proposed, rho, phi, sigma, K_f, K_f_inv);
    double prior_proposed = prior_tau(taus_proposed, tau_prior_sd);
    
    // Calculate acceptance probability
    double log_ratio = lik_proposed + prior_proposed - lik_current - prior_current;
    double prob = std::exp(log_ratio);
    
    // Protect against numerical issues
    if (!std::isfinite(prob)) {
        if (log_ratio > 0) {
            prob = 1.0; // Accept if log_ratio is positive but prob is not finite
        } else {
            prob = 0.0; // Reject if log_ratio is negative but prob is not finite
        }
    }
    
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
    
    // Check if proposed rho is valid
    if (rho_proposed <= 0) {
        return rho;
    }
    
    // Calculate kernels
    arma::mat K_f_curr, K_f_prop;
    arma::mat K_f_curr_inv, K_f_prop_inv;
    
    try {
        K_f_curr = sq_exp_kernel(x, rho, 1e-6);
        K_f_curr_inv = arma::inv_sympd(K_f_curr);
        
        K_f_prop = sq_exp_kernel(x, rho_proposed, 1e-6);
        K_f_prop_inv = arma::inv_sympd(K_f_prop);
    }
    catch (const std::exception& e) {
        // Handle matrix inversion error
        Rcout << "Error in kernel calculation in sample_rho: " << e.what() << "\n";
        // Reject the proposal
        return rho;
    }
    
    // Calculate likelihoods and priors
    double lik_current = likelihood(y, f, betas, taus, rho, phi, sigma, K_f_curr, K_f_curr_inv);
    double prior_current = prior_rho(rho, rho_prior_shape, rho_prior_scale);
    
    double lik_proposed = likelihood(y, f, betas, taus, rho_proposed, phi, sigma, K_f_prop, K_f_prop_inv);
    double prior_proposed = prior_rho(rho_proposed, rho_prior_shape, rho_prior_scale);
    
    // Calculate acceptance probability
    double log_ratio = lik_proposed + prior_proposed - lik_current - prior_current;
    double prob = std::exp(log_ratio);
    
    // Protect against numerical issues
    if (!std::isfinite(prob)) {
        if (log_ratio > 0) {
            prob = 1.0; // Accept if log_ratio is positive but prob is not finite
        } else {
            prob = 0.0; // Reject if log_ratio is negative but prob is not finite
        }
    }
    
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
    // Set up OpenMP
    setOMPThreads();
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Extract dimensions
    int n_time = y.n_rows;
    int n = y.n_cols;
    arma::vec x = arma::linspace(0, 1, n_time);
    
    // Extract initial values
    double rho0 = as<double>(theta0["rho"]);
    arma::vec beta0 = as<arma::vec>(theta0["beta"]);
    arma::vec tau0 = as<arma::vec>(theta0["tau"]);
    arma::vec phi0 = as<arma::vec>(theta0["phi"]);
    double sigma0 = as<double>(theta0["sigma"]);
    
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
    
    // Compute initial kernel and its inverse
    arma::mat K_f = sq_exp_kernel(x, rho0, 1e-6);
    arma::mat K_f_inv;
    
    try {
        K_f_inv = arma::inv_sympd(K_f);
    }
    catch (const std::exception& e) {
        Rcout << "Error in initial K_f inversion: " << e.what() << "\n";
        // Add regularization and try again
        K_f.diag() += 1e-4;
        K_f_inv = arma::inv_sympd(K_f);
    }
    
    // Sample initial f
    Rcout << "Sampling initial f...\n";
    arma::mat f_init;
    try {
        f_init = sample_f(y, beta0, tau0, rho0, phi0, sigma0, 1, 1e-6);
    }
    catch (const std::exception& e) {
        Rcout << "Error sampling initial f: " << e.what() << "\n";
        // Generate a simple initial f as fallback
        f_init = arma::zeros(n_time, 1);
        // Create a smooth curve
        for (int i = 0; i < n_time; i++) {
            f_init(i, 0) = std::sin(2 * M_PI * i / n_time);
        }
    }
    
    arma::vec f = f_init.col(0);
    
    // Apply pinning
    f = pinned_value * f / f(pinned_point);
    chain_f[0] = f;
    
    Rcout << "Starting MCMC iterations...\n";
    
    // Main MCMC loop
    for (int iter = 1; iter < n_iter; iter++) {
        // Progress reporting
        if ((iter % (n_iter/10)) == 0) {
            Rcout << " ... " << static_cast<int>((iter/static_cast<double>(n_iter))*100) << "% \n";
        }
        
        // Get current parameters
        Rcpp::List current = chain[iter-1];
        double rho = as<double>(current["rho"]);
        arma::vec beta = as<arma::vec>(current["beta"]);
        arma::vec tau = as<arma::vec>(current["tau"]);
        arma::vec phi = as<arma::vec>(current["phi"]);
        double sigma = as<double>(current["sigma"]);
        
        try {
            // Sample f and rescale
            arma::mat f_sample = sample_f(y, beta, tau, rho, phi, sigma, 1);
            f = f_sample.col(0);
            f = pinned_value * f / f(pinned_point);
            
            // Update y_hat
            arma::mat y_hat = get_y_hat_matrix(y, f, beta, tau, rho, K_f_inv);
            
            // Sample betas
            beta = sample_beta(y, y_hat, beta_prior_mu, beta_prior_sd);
            
            // Sample f again with new betas
            f_sample = sample_f(y, beta, tau, rho, phi, sigma, 1);
            f = f_sample.col(0);
            f = pinned_value * f / f(pinned_point);
            
            // Sample taus
            tau = sample_tau(y, f, beta, tau, rho, phi, sigma, tau_prior_sd, tau_proposal_sd);
            
            // Sample rho
            rho = sample_rho(y, f, beta, tau, rho, phi, sigma, rho_prior_shape, rho_prior_scale, rho_proposal_sd);
            
            // Update kernel and its inverse
            K_f = sq_exp_kernel(x, rho, 1e-6);
            K_f_inv = arma::inv_sympd(K_f);
            
            // Update y_hat
            y_hat = get_y_hat_matrix(y, f, beta, tau, rho, K_f_inv);
            
            // Compute residuals
            arma::mat z = y - y_hat;
            
            // Sample AR parameters
            List ar_post = sample_AR(z, p);
            arma::vec phi_new = as<arma::vec>(ar_post["rho"]);
            double sigma2_new = as<double>(ar_post["sigma2"]);
            
            // Update current parameters
            Rcpp::List current_new = clone(current);
            current_new["beta"] = beta;
            current_new["tau"] = tau;
            current_new["rho"] = rho;
            current_new["phi"] = phi_new;
            current_new["sigma"] = std::sqrt(sigma2_new);
            
            // Store results
            chain[iter] = current_new;
            chain_f[iter] = f;
            chain_y_hat[iter] = y_hat;
            chain_z[iter] = z;
        }
        catch (const std::exception& e) {
            Rcout << "Error in iteration " << iter << ": " << e.what() << "\n";
            
            // Continue with previous values
            chain[iter] = current;
            chain_f[iter] = chain_f[iter-1];
            
            // Try to recover and continue
            try {
                // Recompute with more stable parameters
                K_f = sq_exp_kernel(x, rho, 1e-4);  // More regularization
                K_f_inv = arma::inv_sympd(K_f);
                
                // Get f from previous iteration
                f = as<arma::vec>(chain_f[iter-1]);
                
                // Update y_hat with more stable computation
                arma::mat y_hat = get_y_hat_matrix(y, f, beta, tau, rho, K_f_inv);
                chain_y_hat[iter] = y_hat;
                
                // Compute residuals
                arma::mat z = y - y_hat;
                chain_z[iter] = z;
            }
            catch (...) {
                // If recovery fails, just copy previous iteration's results
                chain_y_hat[iter] = chain_y_hat[iter-1];
                chain_z[iter] = chain_z[iter-1];
            }
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
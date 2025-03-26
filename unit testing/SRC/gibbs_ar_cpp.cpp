// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>

// Enable C++11
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;

// Forward declarations of helper functions
arma::vec pacf2AR(const arma::vec& pacf);
arma::vec genEpsARMAC(const arma::vec& data, const arma::vec& ar, const arma::vec& ma);
bool is_stationary(const arma::vec& pacf);

// Main GIBBS AR function
// [[Rcpp::export]]
List gibbs_ar_cpp(const arma::vec& z_flat, int ar_order, int Ntotal, int burnin, int seed = 0) {
    // Set random seed if provided
    if (seed != 0) {
        Rcpp::Environment base_env("package:base");
        Rcpp::Function set_seed_r = base_env["set.seed"];
        set_seed_r(seed);
    }
    
    // Input validation
    if (ar_order < 0) {
        Rcpp::stop("'ar_order' must be >= 0");
    }
    if (Ntotal <= 0) {
        Rcpp::stop("'Ntotal' must be > 0");
    }
    if (burnin < 0 || burnin >= Ntotal) {
        Rcpp::stop("'burnin' must be >= 0 and < Ntotal");
    }
    
    // Parameter settings that match the defaults in the R function
    int thin = 1;
    int print_interval = 500;
    double numerical_thresh = 1e-7;
    int adaption_N = burnin;
    int adaption_batchSize = 50;
    double adaption_tar = 0.44;
    bool full_lik = false;
    
    // Prior parameters
    arma::vec rho_alpha(ar_order, arma::fill::ones);
    arma::vec rho_beta(ar_order, arma::fill::ones);
    double sigma2_alpha = 0.001;
    double sigma2_beta = 0.001;
    
    // Center the data
    arma::vec z = z_flat - arma::mean(z_flat);
    int n = z.n_elem;
    
    // Initialize storage for MCMC samples
    arma::mat rho_trace(ar_order, Ntotal, arma::fill::zeros);
    arma::vec sigma2_trace(Ntotal, arma::fill::zeros);
    arma::vec lpostTrace(Ntotal, arma::fill::zeros);
    arma::vec deviance(Ntotal, arma::fill::zeros);
    
    // Initialize AR parameters
    if (ar_order > 0) {
        // Estimate initial AR model using Yule-Walker equations
        arma::vec r(ar_order + 1);
        r(0) = 1.0;
        
        // Calculate autocorrelations
        for (int h = 1; h <= ar_order; h++) {
            double sum = 0.0;
            for (int t = 0; t < n - h; t++) {
                sum += z(t) * z(t + h);
            }
            r(h) = sum / (n - h);
        }
        
        // Normalize autocorrelations
        r = r / r(0);
        
        // Set initial PACF values
        if (ar_order >= 1) rho_trace(0, 0) = r(1);
        if (ar_order >= 2) rho_trace(1, 0) = (r(2) - r(1)*r(1)) / (1 - r(1)*r(1));
        
        // Fill in additional PACF values if needed
        for (int j = 2; j < ar_order; j++) {
            rho_trace(j, 0) = 0.1;  // Default value
        }
        
        // Ensure PACF values are in [-1, 1]
        for (int j = 0; j < ar_order; j++) {
            if (std::abs(rho_trace(j, 0)) >= 1.0) {
                rho_trace(j, 0) = 0.95 * rho_trace(j, 0) / std::abs(rho_trace(j, 0));
            }
        }
        
        // Convert PACF to AR coefficients
        arma::vec ar_coefs = pacf2AR(rho_trace.col(0));
        
        // Generate residuals for initial sigma2 estimate
        arma::vec eps = genEpsARMAC(z, ar_coefs, arma::vec());
        
        // Calculate initial sigma2
        double sum_sq = 0.0;
        for (int t = ar_order; t < n; t++) {
            sum_sq += std::pow(eps(t), 2);
        }
        sigma2_trace(0) = sum_sq / (n - ar_order);
    } else {
        // For AR(0), just use sample variance
        sigma2_trace(0) = arma::var(z);
    }
    
    // Make sure sigma2 is positive
    if (sigma2_trace(0) <= 0) sigma2_trace(0) = 0.01;
    
    // Proposal variance for PACF parameters
    arma::vec var_prop_rho(ar_order, arma::fill::ones);
    var_prop_rho = var_prop_rho * 0.1;  // Initial value
    
    // Compute log-posterior for initial state
    double f_store = 0.0;
    if (ar_order > 0) {
        arma::vec a = pacf2AR(rho_trace.col(0));
        arma::vec eps = genEpsARMAC(z, a, arma::vec());
        
        // Calculate log-posterior
        double logprior_rho = 0.0;
        for (int j = 0; j < ar_order; j++) {
            // Beta(alpha, beta) prior for transformed PACF
            double rho_j = 0.5 * (rho_trace(j, 0) + 1.0);  // Map from [-1,1] to [0,1]
            logprior_rho += (rho_alpha(j) - 1.0) * std::log(rho_j) + 
                           (rho_beta(j) - 1.0) * std::log(1.0 - rho_j);
        }
        
        // Log-likelihood for AR model
        double log_like = -0.5 * (n - ar_order) * std::log(2.0 * M_PI * sigma2_trace(0)) - 
                         0.5 * sum(arma::square(eps.subvec(ar_order, n-1))) / sigma2_trace(0);
        
        // Log-prior for sigma2 (inverse gamma)
        double logprior_sigma2 = -sigma2_alpha * std::log(sigma2_trace(0)) - 
                                sigma2_beta / sigma2_trace(0);
        
        f_store = log_like + logprior_rho + logprior_sigma2;
        lpostTrace(0) = f_store;
        
        // Deviance = -2 * log-likelihood
        deviance(0) = -2 * log_like;
    }
    
    // Main MCMC loop
    for (int j = 1; j < Ntotal; j++) {
        // Print progress
        if (j % print_interval == 0) {
            Rcpp::Rcout << "Iteration " << j << " of " << Ntotal 
                       << " (" << (100.0 * j / Ntotal) << "%)\n";
        }
        
        // Adapt proposal distribution if in adaptation phase
        if (ar_order > 0 && j < adaption_N && j > 1 && j % adaption_batchSize == 1) {
            // Calculate acceptance rates for batch
            arma::mat batch_rho = rho_trace.cols(j - adaption_batchSize, j - 1);
            arma::vec acceptance_rates(ar_order);
            
            for (int p = 0; p < ar_order; p++) {
                int changes = 0;
                for (int i = 1; i < adaption_batchSize; i++) {
                    if (batch_rho(p, i) != batch_rho(p, i-1)) {
                        changes++;
                    }
                }
                acceptance_rates(p) = static_cast<double>(changes) / (adaption_batchSize - 1);
            }
            
            // Adjust proposal variances
            double adaption_delta = std::min(0.1, 1.0 / std::pow(j, 1.0/3.0));
            
            for (int p = 0; p < ar_order; p++) {
                double adjustment = ((acceptance_rates(p) > adaption_tar) ? 1.0 : -1.0) * adaption_delta;
                var_prop_rho(p) = var_prop_rho(p) * std::exp(2.0 * adjustment);
            }
        }
        
        // Copy previous values
        rho_trace.col(j) = rho_trace.col(j-1);
        sigma2_trace(j) = sigma2_trace(j-1);
        
        // Update PACF coefficients with Metropolis steps
        if (ar_order > 0) {
            arma::vec current_rho = rho_trace.col(j);
            arma::vec a = pacf2AR(current_rho);  // Current AR coefficients
            
            for (int p = 0; p < ar_order; p++) {
                bool rejected = false;
                arma::vec rho_star = current_rho;
                
                // Propose new PACF value - mix of normal and uniform proposals
                if (R::rbinom(1, 0.75) == 1) {
                    // Normal proposal
                    rho_star(p) = current_rho(p) + R::rnorm(0, std::sqrt(var_prop_rho(p)));
                    if (std::abs(rho_star(p)) >= 1.0) {
                        rejected = true;
                    }
                } else {
                    // Uniform proposal
                    rho_star(p) = R::runif(-1.0, 1.0);
                }
                
                if (!rejected) {
                    // Check if process is stationary
                    if (!is_stationary(rho_star)) {
                        rejected = true;
                    } else {
                        // Calculate new AR coefficients
                        arma::vec a_star = pacf2AR(rho_star);
                        
                        // Generate residuals
                        arma::vec eps = genEpsARMAC(z, a, arma::vec());
                        arma::vec eps_star = genEpsARMAC(z, a_star, arma::vec());
                        
                        // Calculate sum of squared residuals
                        double sum_sq = 0.0;
                        double sum_sq_star = 0.0;
                        for (int t = ar_order; t < n; t++) {
                            sum_sq += std::pow(eps(t), 2);
                            sum_sq_star += std::pow(eps_star(t), 2);
                        }
                        
                        // Calculate log-posterior components
                        double log_ratio = -0.5 * (sum_sq_star - sum_sq) / sigma2_trace(j);
                        
                        // Add prior ratio if using informative priors
                        if (arma::any(rho_alpha != 1.0) || arma::any(rho_beta != 1.0)) {
                            // Calculate prior ratio
                            double logprior_current = 0.0;
                            double logprior_proposed = 0.0;
                            
                            for (int k = 0; k < ar_order; k++) {
                                // Only consider the coefficient being updated
                                if (k == p) {
                                    double rho_current = 0.5 * (current_rho(k) + 1.0);
                                    double rho_proposed = 0.5 * (rho_star(k) + 1.0);
                                    
                                    logprior_current += (rho_alpha(k) - 1.0) * std::log(rho_current) + 
                                                      (rho_beta(k) - 1.0) * std::log(1.0 - rho_current);
                                    logprior_proposed += (rho_alpha(k) - 1.0) * std::log(rho_proposed) + 
                                                       (rho_beta(k) - 1.0) * std::log(1.0 - rho_proposed);
                                }
                            }
                            
                            log_ratio += logprior_proposed - logprior_current;
                        }
                        
                        // Metropolis acceptance
                        if (std::log(R::runif(0, 1)) < log_ratio) {
                            current_rho = rho_star;
                            a = a_star;
                        }
                    }
                }
            }
            
            // Store updated PACF values
            rho_trace.col(j) = current_rho;
            
            // Update sigma2 by sampling from inverse gamma
            arma::vec eps = genEpsARMAC(z, a, arma::vec());
            double sum_sq = 0.0;
            for (int t = ar_order; t < n; t++) {
                sum_sq += std::pow(eps(t), 2);
            }
            
            // Posterior shape and rate for inverse gamma
            double alpha_post = sigma2_alpha + (n - ar_order) / 2.0;
            double beta_post = sigma2_beta + sum_sq / 2.0;
            
            // Sample from inverse gamma by sampling from gamma and taking reciprocal
            double gamma_sample = R::rgamma(alpha_post, 1.0 / beta_post);
            sigma2_trace(j) = 1.0 / gamma_sample;
            
            // Calculate log-posterior and deviance
            double log_like = -0.5 * (n - ar_order) * std::log(2.0 * M_PI * sigma2_trace(j)) - 
                             0.5 * sum_sq / sigma2_trace(j);
            
            double logprior_rho = 0.0;
            for (int k = 0; k < ar_order; k++) {
                double rho_k = 0.5 * (current_rho(k) + 1.0);
                logprior_rho += (rho_alpha(k) - 1.0) * std::log(rho_k) + 
                               (rho_beta(k) - 1.0) * std::log(1.0 - rho_k);
            }
            
            double logprior_sigma2 = -sigma2_alpha * std::log(sigma2_trace(j)) - 
                                    sigma2_beta / sigma2_trace(j);
            
            lpostTrace(j) = log_like + logprior_rho + logprior_sigma2;
            deviance(j) = -2 * log_like;
        } else {
            // For AR(0), just sample sigma2 directly
            double alpha_post = sigma2_alpha + n / 2.0;
            double beta_post = sigma2_beta + arma::sum(arma::square(z)) / 2.0;
            double gamma_sample = R::rgamma(alpha_post, 1.0 / beta_post);
            sigma2_trace(j) = 1.0 / gamma_sample;
            
            // Calculate log-posterior and deviance
            double log_like = -0.5 * n * std::log(2.0 * M_PI * sigma2_trace(j)) - 
                             0.5 * arma::sum(arma::square(z)) / sigma2_trace(j);
            
            double logprior_sigma2 = -sigma2_alpha * std::log(sigma2_trace(j)) - 
                                    sigma2_beta / sigma2_trace(j);
            
            lpostTrace(j) = log_like + logprior_sigma2;
            deviance(j) = -2 * log_like;
        }
    }
    
    // Extract posterior samples (after burnin)
    arma::mat rho_samples = rho_trace.cols(burnin, Ntotal-1);
    arma::vec sigma2_samples = sigma2_trace.subvec(burnin, Ntotal-1);
    arma::vec lpost_samples = lpostTrace.subvec(burnin, Ntotal-1);
    
    // Return results in the same format as the R function
    return Rcpp::List::create(
        Rcpp::Named("rho") = rho_samples.t(),   // Transpose to match R output
        Rcpp::Named("sigma2") = sigma2_samples
    );
}

// Helper function to convert PACF to AR coefficients using Durbin-Levinson
arma::vec pacf2AR(const arma::vec& pacf) {
    int p = pacf.n_elem;
    
    if (p == 0) return arma::vec();
    if (p == 1) return pacf;
    
    // For AR(p > 1), use Durbin-Levinson recursion
    arma::mat phi(p, p, arma::fill::zeros);
    phi(0, 0) = pacf(0);
    
    for (int k = 1; k < p; k++) {
        phi(k, k) = pacf(k);
        for (int j = 0; j < k; j++) {
            phi(k, j) = phi(k-1, j) - pacf(k) * phi(k-1, k-1-j);
        }
    }
    
    // Return last row
    return phi.row(p-1).t();
}

// Generate residuals from AR model
arma::vec genEpsARMAC(const arma::vec& data, const arma::vec& ar, const arma::vec& ma) {
    int n = data.n_elem;
    int p = ar.n_elem;
    int q = ma.n_elem;
    arma::vec eps(n, arma::fill::zeros);
    
    // Handle initial values
    for (int t = 0; t < std::max(p, q); t++) {
        eps(t) = data(t);  // Approximation for initial values
    }
    
    // Generate remaining residuals
    for (int t = p; t < n; t++) {
        double pred = 0.0;
        for (int j = 0; j < p; j++) {
            pred += ar(j) * data(t-j-1);
        }
        for (int j = 0; j < std::min(q, t); j++) {
            pred += ma(j) * eps(t-j-1);
        }
        eps(t) = data(t) - pred;
    }
    
    return eps;
}

// Check if AR coefficients represent a stationary process
bool is_stationary(const arma::vec& pacf) {
    // First check if all PACF values are in [-1, 1]
    for (size_t i = 0; i < pacf.n_elem; i++) {
        if (std::abs(pacf(i)) >= 1.0) {
            return false;
        }
    }
    
    // For AR(1), that's all we need to check
    if (pacf.n_elem <= 1) return true;
    
    // For AR(p > 1), convert to AR coefficients and check eigenvalues
    arma::vec ar = pacf2AR(pacf);
    int p = ar.n_elem;
    
    // Create companion matrix
    arma::mat A = arma::zeros(p, p);
    for (int i = 0; i < p; i++) {
        A(0, i) = ar(i);
    }
    for (int i = 1; i < p; i++) {
        A(i, i-1) = 1.0;
    }
    
    // Check eigenvalues
    try {
        arma::cx_vec eigval = arma::eig_gen(A);
        for (int i = 0; i < p; i++) {
            if (std::abs(eigval(i)) >= 1.0) {
                return false;
            }
        }
    } catch (...) {
        // If eigenvalue calculation fails, consider it non-stationary
        return false;
    }
    
    return true;
}
// Matrix Utility Functions 

// Convert upper triangular to symmetric matrix
// @param m Upper triangular matrix
// [[Rcpp::export]]
arma::mat ultosymmetric_cpp(const arma::mat& m) {
    return m + m.t() - diagmat(m.diag());
}

// Calculate AR process covariance matrix
// @param phi AR coefficients
// @param sigma Error standard deviation
// @param n_time Number of time points
// [[Rcpp::export]]
arma::mat get_Sigma_nu_cpp(const arma::vec& phi, double sigma, int n_time) {
    // Create a basic AR(p) covariance structure
    const int p = phi.n_elem;
    arma::mat Sigma(n_time, n_time, arma::fill::zeros);
    
    // Compute autocovariances - improved stability
    arma::vec acf = arma::vec(n_time, arma::fill::zeros);
    const double sigma_sq = sigma * sigma;
    const double phi_squared_sum = arma::sum(phi % phi);
    
    // Avoid division by zero or near-zero
    if (phi_squared_sum >= 1.0) {
        acf(0) = sigma_sq * 100.0; // Large variance for nearly non-stationary processes
    } else {
        acf(0) = sigma_sq / (1.0 - phi_squared_sum);
    }
    
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
    
    // Ensure symmetry
    Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0;
    return Sigma_i;
}
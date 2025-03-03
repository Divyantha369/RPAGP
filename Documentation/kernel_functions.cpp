// Kernel Functions 

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
    
    // Optimize by precomputing x values
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
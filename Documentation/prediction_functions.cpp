// Prediction Functions

// Get y_hat for a single trial
// @param f Vector of f values
// @param K_i Covariance matrix K_i
// @param K_f_inv Inverse of covariance matrix of f
// [[Rcpp::export]]
arma::vec get_y_hat_single_cpp(const arma::vec& f, const arma::mat& K_i, const arma::mat& K_f_inv) {
    // Optimize the order of matrix multiplications
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
    
    // Pre-compute K_f_inv * f once, as it's used for all trials
    const arma::vec f_transformed = K_f_inv * f;
    
    // Parallelize over trials if enough of them
    #pragma omp parallel for if(n > 8)
    for(int i = 0; i < n; i++) {
        // Use cached K_i matrix
        arma::mat K_i = cached_get_K_i_cpp(x, rho, tau[i], beta[i]);
        
        // Compute y_hat using pre-computed f_transformed
        y_hat.col(i) = K_i * f_transformed;
    }
    
    return y_hat;
}
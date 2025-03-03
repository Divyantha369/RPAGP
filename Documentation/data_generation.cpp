// Data Generation Functions 

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
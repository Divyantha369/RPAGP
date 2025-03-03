// Multivariate normal density calculation 
// @param x Vector of observations
// @param mean Mean vector
// @param sigma Covariance matrix
// [[Rcpp::export]]
double dmvnorm_cpp(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma) {
    const int n = x.n_elem;
    double log_det_val;
    double sign;
    
    // Cholesky decomposition 
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
            return -1e10; 
        }
        
        arma::vec x_centered = x - mean;
        
        double quadform;
        bool solve_success = arma::solve(quadform, sigma, x_centered, x_centered);
        
        if (!solve_success) {
    
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
    
    // Precompute matrices for each trial 
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
        // Compute multivariate normal log density
        tmp += dmvnorm_cpp(y.col(i), mu_storage[i], Sigma_storage[i]);
    }
    
    // Handle extreme negative values
    if (tmp < -1e9) {
        return -1e10;
    }
    
    return tmp;
}
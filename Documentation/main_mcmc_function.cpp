// Main MCMC Fitting Function

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
    
    // Initialize 
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
    
    // Pre-allocate y_hat matrix to avoid repeated allocation
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
        
        // Get y_hat 
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
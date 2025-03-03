// MCMC Sampling Functions

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
// This approach avoids explicit mutex locks by using a reduction
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
    
    // Pre-allocate matrices to avoid repeated allocations
    arma::mat A(n_time, n_time);
    arma::vec b(n_time);
    
    // Pre-compute K_i matrices and other reusable components
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
        // Initialize A and b
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
        
        // Ensure A is symmetric for numerical stability
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

// AR simulation for residuals (simplified)
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
    
    // Flatten the residuals - pre-allocate
    arma::vec z_vec(n_time * n);
    for (int i = 0; i < n; i++) {
        z_vec.subvec(i * n_time, (i + 1) * n_time - 1) = z.col(i);
    }
    
    // Set up the design matrix X for the AR model - pre-allocate
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
    
    // Sample phi - QR decomposition
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
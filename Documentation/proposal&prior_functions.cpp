// MCMC Proposals and Priors

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
    
    // Compute log density of multivariate normal
    const arma::vec mean = arma::zeros(n);
    double log_det_val;
    double sign;
    arma::log_det(log_det_val, sign, Sigma);
    
    if (sign <= 0) {
        return -1e10;
    }
    
    // Try to solve directly instead of using inverse
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
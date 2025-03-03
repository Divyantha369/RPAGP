// Caching Structures 

// Cache key structures
struct KernelKey {
    int n;
    double rho;
    double alpha;
    double nugget;
    
    bool operator<(const KernelKey& other) const {
        if (n != other.n) return n < other.n;
        if (rho != other.rho) return rho < other.rho;
        if (alpha != other.alpha) return alpha < other.alpha;
        return nugget < other.nugget;
    }
};

struct KiKey {
    int n;
    double rho;
    double tau;
    double beta;
    
    bool operator<(const KiKey& other) const {
        if (n != other.n) return n < other.n;
        if (rho != other.rho) return rho < other.rho;
        if (tau != other.tau) return tau < other.tau;
        return beta < other.beta;
    }
};

struct SigmaNuKey {
    int n;
    double sigma;
    std::string phi_str;
    
    bool operator<(const SigmaNuKey& other) const {
        if (n != other.n) return n < other.n;
        if (sigma != other.sigma) return sigma < other.sigma;
        return phi_str < other.phi_str;
    }
};

// Global caches for expensive matrix computations
std::map<KernelKey, arma::mat> kernel_cache;
std::map<KiKey, arma::mat> k_i_cache;
std::map<SigmaNuKey, arma::mat> sigma_nu_cache;

// Cache size limits to prevent memory bloat
const size_t MAX_KERNEL_CACHE_SIZE = 30;
const size_t MAX_KI_CACHE_SIZE = 200;
const size_t MAX_SIGMA_NU_CACHE_SIZE = 30;

// Helper function to convert phi vector to string for caching
std::string phi_to_string(const arma::vec& phi) {
    std::ostringstream ss;
    for (size_t i = 0; i < phi.n_elem; i++) {
        ss << phi(i);
        if (i < phi.n_elem - 1) ss << ",";
    }
    return ss.str();
}

// Clear all caches
void clear_caches() {
    kernel_cache.clear();
    k_i_cache.clear();
    sigma_nu_cache.clear();
}

// Clear caches and reset function
// [[Rcpp::export]]
void reset_caches_cpp() {
    clear_caches();
    Rcpp::Rcout << "All computation caches have been cleared.\n";
}
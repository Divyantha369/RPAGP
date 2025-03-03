# RPAGP: Reduced Preferential Attachment Gaussian Process

![optimized-rpagp-flowchart](https://github.com/user-attachments/assets/92f64c4c-4c2b-44ee-8961-a009d4940964)

## Implementation Details

This repository contains a high-performance C++ implementation of the Reduced Preferential Attachment Gaussian Process (RPAGP) model for MCMC sampling. The implementation leverages advanced caching mechanisms and parallelization techniques to efficiently handle complex time series analysis.

### Performance Optimization

#### Caching System
The implementation utilizes three specialized caching structures to avoid redundant matrix computations:

- **Kernel Matrix Cache**: Stores square exponential kernel matrices with a configurable maximum size of 30 entries
- **K_i Matrix Cache**: Stores trial-specific covariance matrices with a larger capacity (200 entries)
- **Sigma_nu Cache**: Stores AR process covariance matrices

Each cache uses custom key structures based on relevant parameters (dimensions, hyperparameters) with efficient lookup mechanisms.

```cpp
// Example of cached kernel computation
arma::mat cached_sq_exp_kernel_cpp(const arma::vec& x, double rho, double alpha, double nugget)

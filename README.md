# BayesRPAGP C++ Implementation with Parallelization

![optimized-rpagp-flowchart](https://github.com/user-attachments/assets/92f64c4c-4c2b-44ee-8961-a009d4940964)

## Implementation Details

This repository contains a  C++ implementation of the Random Phase-Amplitude Gaussian Process (RPAGP) algorithm for Bayesian inference of trial-level amplitude, latency, and ERP waveforms. The implementation leverages caching mechanisms and parallelization techniques to handle complex time series analysis efficiently.

### Performance Optimization

#### Caching System
The implementation utilizes three specialized caching structures to avoid redundant matrix computations:

- **Kernel Matrix Cache**: Stores square exponential kernel matrices with a configurable maximum size of 30 entries
- **K_i Matrix Cache**: Stores trial-specific covariance matrices with a larger capacity (200 entries)
- **Sigma_nu Cache**: Stores AR process covariance matrices

Each cache uses custom key structures based on relevant parameters (dimensions, hyperparameters) with  lookup mechanisms.

#### Parallelization Strategy
The implementation employs OpenMP for multi-threaded execution:

- **Likelihood Computation**: Distributes trial-specific calculations across available cores
- **Parameter Sampling**: Parallelizes sampling of beta parameters and latent functions
- **Matrix Operations**: Concurrent generation of predicted values and synthetic data
- **Adaptive Threading**: Automatically determines optimal thread count based on system resources

#### Computational Efficiency
Additional optimizations include:

- **Matrix Reuse**: Pre-computes matrices that remain constant across iterations
- **Numerical Stability**: Employs Cholesky decomposition with fallbacks for matrix operations
- **Memory Management**: cache clearing to prevent memory bloat during long MCMC runs

### MCMC Algorithm
The MCMC sampling procedure iterates through:

1. Sampling the latent function with appropriate pinning constraints
2. Sampling trial-specific amplitudes (beta)
3. Sampling latency parameters (tau) via Metropolis-Hastings
4. Sampling the length scale parameter (rho) via Metropolis-Hastings
5. Sampling AR process parameters for the noise model


![plot](https://github.com/user-attachments/assets/c1cbd19b-b871-4419-b76f-7e64949f2a6b)


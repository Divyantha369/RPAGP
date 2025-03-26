# BayesRPAGP C++ Implementation with Parallelization

![optimized-rpagp-flowchart](https://github.com/user-attachments/assets/92f64c4c-4c2b-44ee-8961-a009d4940964)

## Implementation Details

This repository contains a  C++ implementation of the Random Phase-Amplitude Gaussian Process (RPAGP) algorithm for Bayesian inference of trial-level amplitude, latency, and ERP waveforms. The implementation leverages caching mechanisms and parallelization techniques to handle complex time series analysis efficiently.

### Performance Optimization


#### Parallelization Strategy
The implementation employs OpenMP for multi-threaded execution:

- **Likelihood Computation**: Distributes trial-specific calculations across available cores
- **Parameter Sampling**: Parallelizes sampling of beta parameters and latent functions
- **Matrix Operations**: Concurrent generation of predicted values 
- **Adaptive Threading**: Automatically determines optimal thread count based on system resources

#### Use of Armadillo for efficent Operations

Armadillo underpins the RPAGP code’s linear algebra, providing a clean, high-level interface to BLAS/LAPACK functionality. By handling matrix and vector operations through Armadillo’s expression templates, we minimize overhead and memory copies. This is further boosted by OpenMP parallelization, which distributes computations across multiple cores. Overall, Armadillo greatly simplifies the implementation while delivering efficient performance for the code’s frequent matrix computations.



### MCMC Algorithm
The MCMC sampling procedure iterates through:

1. Sampling the latent function 
2. Sampling trial-specific amplitudes (beta)
3. Sampling latency parameters (tau) via Metropolis-Hastings
4. Sampling the length scale parameter (rho) via Metropolis-Hastings
5. Sampling AR process parameters for the noise model


![plot](https://github.com/user-attachments/assets/c1cbd19b-b871-4419-b76f-7e64949f2a6b)


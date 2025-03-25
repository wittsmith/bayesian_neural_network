#ifndef PRIOR_H
#define PRIOR_H

// Common interface for prior distributions in a Bayesian neural network layer.
typedef struct Prior {
    void *data; // Pointer to prior-specific parameters
    // Function pointer to compute the KL divergence for a given variational posterior parameter.
    // Given mu (mean) and logvar (log variance) of the variational posterior,
    // returns the KL divergence (or an approximation thereof) for that parameter.
    double (*compute_kl)(struct Prior *prior, double mu, double logvar);
    
    // Function pointer to compute the log-probability of a value under the prior.
    double (*log_prob)(struct Prior *prior, double x);
} Prior;

#endif // PRIOR_H

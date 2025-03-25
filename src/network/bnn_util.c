#include "bnn_util.h"
#include "random_utils.h"  // For random_gaussian()
#include <math.h>

// sample_gaussian:
// Uses the reparameterization trick: sample = mean + exp(0.5 * logvar) * epsilon,
// with epsilon drawn from a standard normal distribution.
double sample_gaussian(double mean, double logvar) {
    double stddev = exp(0.5 * logvar);
    double epsilon = random_gaussian(0.0, 1.0);
    return mean + stddev * epsilon;
}

// kl_divergence_single:
// Computes the KL divergence between N(mu, exp(logvar)) and N(0, prior_variance).
double kl_divergence_single(double mu, double logvar, double prior_variance) {
    double sigma2 = exp(logvar);
    // KL divergence: 0.5 * ( (sigma^2 + mu^2) / prior_variance - 1 + log(prior_variance) - logvar )
    return 0.5 * ((sigma2 + mu * mu) / prior_variance - 1.0 + log(prior_variance) - logvar);
}

// compute_total_kl_divergence:
// Sums the KL divergence for each element in the arrays mu and logvar.
double compute_total_kl_divergence(const double *mu, const double *logvar, int length, double prior_variance) {
    double total_kl = 0.0;
    for (int i = 0; i < length; i++) {
        total_kl += kl_divergence_single(mu[i], logvar[i], prior_variance);
    }
    return total_kl;
}

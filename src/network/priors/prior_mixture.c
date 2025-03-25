#include "prior_mixture.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Define a structure to hold parameters for a two-component mixture-of-Gaussians.
typedef struct {
    double mu1, sigma1;
    double mu2, sigma2;
    double lambda; // Mixing coefficient for the first component (the weight for the second is 1 - lambda).
} MixturePriorData;

// Compute the log-density for a single Gaussian.
static double gaussian_log_density(double x, double mu, double sigma) {
    return -0.5 * (log(2 * M_PI) + 2 * log(sigma) + pow((x - mu) / sigma, 2));
}

// Compute the log-density of a two-component mixture-of-Gaussians.
// p(x) = lambda * N(x|mu1, sigma1^2) + (1-lambda) * N(x|mu2, sigma2^2)
// For numerical stability, use log-sum-exp.
static double mixture_log_density(double x, MixturePriorData *data) {
    double log_prob1 = log(data->lambda) + gaussian_log_density(x, data->mu1, data->sigma1);
    double log_prob2 = log(1.0 - data->lambda) + gaussian_log_density(x, data->mu2, data->sigma2);
    // Use log-sum-exp:
    double max_log = (log_prob1 > log_prob2) ? log_prob1 : log_prob2;
    return max_log + log(exp(log_prob1 - max_log) + exp(log_prob2 - max_log));
}

// A simple approximation for the KL divergence between a Gaussian variational posterior N(mu, sigma^2)
// and the mixture prior can be done by: KL ≈ E_q[log q(x)] - E_q[log p(x)]
// Here we approximate E_q[log p(x)] by evaluating at mu.
static double kl_divergence_mixture(double mu, double logvar, MixturePriorData *data) {
    double sigma2 = exp(logvar);
    double log_q = -0.5 * log(2 * M_PI * M_E * sigma2);
    double log_p = mixture_log_density(mu, data);
    return log_q - log_p;
}

// Implementation of the KL divergence function pointer for the mixture prior.
static double mixture_compute_kl(Prior *prior, double mu, double logvar) {
    MixturePriorData *data = (MixturePriorData*) prior->data;
    return kl_divergence_mixture(mu, logvar, data);
}

// Implementation of the log-probability function pointer for the mixture prior.
static double mixture_log_prob(Prior *prior, double x) {
    MixturePriorData *data = (MixturePriorData*) prior->data;
    return mixture_log_density(x, data);
}

// Create a mixture-of-Gaussians prior object.
Prior* create_mixture_prior(double mu1, double sigma1, double mu2, double sigma2, double lambda) {
    Prior *prior = (Prior*) malloc(sizeof(Prior));
    if (!prior) {
        fprintf(stderr, "Failed to allocate Mixture prior.\n");
        exit(EXIT_FAILURE);
    }
    MixturePriorData *data = (MixturePriorData*) malloc(sizeof(MixturePriorData));
    if (!data) {
        fprintf(stderr, "Failed to allocate Mixture prior data.\n");
        exit(EXIT_FAILURE);
    }
    data->mu1 = mu1;
    data->sigma1 = sigma1;
    data->mu2 = mu2;
    data->sigma2 = sigma2;
    data->lambda = lambda;
    
    prior->data = data;
    prior->compute_kl = mixture_compute_kl;
    prior->log_prob = mixture_log_prob;
    return prior;
}

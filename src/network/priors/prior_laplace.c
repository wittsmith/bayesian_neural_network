#include "prior_laplace.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Define a structure to hold Laplace-specific parameters.
typedef struct {
    double location; // Typically 0
    double scale;    // Controls the spread
} LaplacePriorData;

// Compute the log-density of a Laplace distribution: 
// log p(x) = -log(2*scale) - |x - location| / scale
static double laplace_log_density(double x, double location, double scale) {
    return -log(2 * scale) - fabs(x - location) / scale;
}

// Approximate KL divergence between a Gaussian variational posterior N(mu, sigma^2)
// (where sigma^2 = exp(logvar)) and a Laplace prior.
// Here we approximate KL(q||p) ≈ E_q[log q(x)] - E_q[log p(x)].
// For simplicity, we approximate E_q[log p(x)] by evaluating at mu.
static double kl_divergence_laplace(double mu, double logvar, double location, double scale) {
    double sigma2 = exp(logvar);
    // For a Gaussian q: E_q[log q(x)] = -0.5 * log(2*pi*e*sigma^2)
    double log_q = -0.5 * log(2 * M_PI * M_E * sigma2);
    // Approximate E_q[log p(x)] by evaluating at the mean (a crude approximation)
    double log_p = laplace_log_density(mu, location, scale);
    return log_q - log_p;
}

// Implementation of the KL divergence function pointer for the Laplace prior.
static double laplace_compute_kl(Prior *prior, double mu, double logvar) {
    LaplacePriorData *data = (LaplacePriorData*) prior->data;
    return kl_divergence_laplace(mu, logvar, data->location, data->scale);
}

// Implementation of the log-probability function pointer for the Laplace prior.
static double laplace_log_prob(Prior *prior, double x) {
    LaplacePriorData *data = (LaplacePriorData*) prior->data;
    return laplace_log_density(x, data->location, data->scale);
}

// Create a Laplace prior object.
Prior* create_laplace_prior(double location, double scale) {
    Prior *prior = (Prior*) malloc(sizeof(Prior));
    if (!prior) {
        fprintf(stderr, "Failed to allocate Laplace prior.\n");
        exit(EXIT_FAILURE);
    }
    LaplacePriorData *data = (LaplacePriorData*) malloc(sizeof(LaplacePriorData));
    if (!data) {
        fprintf(stderr, "Failed to allocate Laplace prior data.\n");
        exit(EXIT_FAILURE);
    }
    data->location = location;
    data->scale = scale;
    
    prior->data = data;
    prior->compute_kl = laplace_compute_kl;
    prior->log_prob = laplace_log_prob;
    return prior;
}

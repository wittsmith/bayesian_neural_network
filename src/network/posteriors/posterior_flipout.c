#include "posterior_flipout.h"
#include "../bnn_util.h"      // For kl_divergence_single()
#include "../utils/utils.h"
#include "../utils/random_utils.h"
#include <stdlib.h>
#include <math.h>

// Flipout sample function:
// Implements a simplified Flipout approach by generating a Rademacher random variable (±1)
// and applying it to the noise sample.
static double flipout_sample(Posterior *posterior, double mu, double logvar) {
    // Generate a Rademacher random variable: +1 or -1.
    int sign = (random_uniform() < 0.5) ? 1 : -1;
    double noise = random_gaussian(0.0, 1.0);
    double std = sqrt(exp(logvar));
    return mu + sign * std * noise;
}

// Flipout KL divergence function: we use the standard Gaussian KL divergence.
static double flipout_compute_kl(Posterior *posterior, double mu, double logvar) {
    return kl_divergence_single(mu, logvar, 1.0); // Default prior variance assumed to be 1.0.
}

Posterior* create_flipout_posterior() {
    Posterior *posterior = (Posterior*) malloc(sizeof(Posterior));
    if (!posterior) {
        handle_error("Failed to allocate Posterior in create_flipout_posterior.");
    }
    // In this simple implementation, no additional data is needed.
    posterior->data = NULL;
    posterior->sample = flipout_sample;
    posterior->compute_kl = flipout_compute_kl;
    return posterior;
}

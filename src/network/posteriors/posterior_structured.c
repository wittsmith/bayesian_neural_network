#include "posterior_structured.h"
#include "../bnn_util.h"      // For sample_gaussian() and kl_divergence_single()
#include "../utils/utils.h"
#include "../utils/random_utils.h"
#include <stdlib.h>
#include <math.h>

// Sample function for structured posterior:
// For demonstration, we sample as: sample = mu + structure_scale * sqrt(exp(logvar)) * noise.
static double structured_sample(Posterior *posterior, double mu, double logvar) {
    StructuredPosteriorData *data = (StructuredPosteriorData*) posterior->data;
    double noise = random_gaussian(0.0, 1.0);
    double std = sqrt(exp(logvar));
    return mu + data->structure_scale * std * noise;
}

// KL divergence function for structured posterior:
// Here we approximate KL divergence as the standard Gaussian KL (with default prior variance 1.0) scaled by structure_scale.
static double structured_compute_kl(Posterior *posterior, double mu, double logvar) {
    StructuredPosteriorData *data = (StructuredPosteriorData*) posterior->data;
    double base_kl = kl_divergence_single(mu, logvar, 1.0); // Assuming a default Gaussian prior with variance 1.0.
    return data->structure_scale * base_kl;
}

Posterior* create_structured_posterior(double structure_scale) {
    Posterior *posterior = (Posterior*) malloc(sizeof(Posterior));
    if (!posterior) {
        handle_error("Failed to allocate Posterior in create_structured_posterior.");
    }
    StructuredPosteriorData *data = (StructuredPosteriorData*) malloc(sizeof(StructuredPosteriorData));
    if (!data) {
        handle_error("Failed to allocate StructuredPosteriorData in create_structured_posterior.");
    }
    data->structure_scale = structure_scale;
    
    posterior->data = data;
    posterior->sample = structured_sample;
    posterior->compute_kl = structured_compute_kl;
    
    return posterior;
}

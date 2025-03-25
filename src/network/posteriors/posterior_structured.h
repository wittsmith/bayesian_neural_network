#ifndef POSTERIOR_STRUCTURED_H
#define POSTERIOR_STRUCTURED_H

#include "posterior.h"  // Use the centralized Posterior interface

// Structured posterior data structure.
// For demonstration, we use a single scaling parameter to mimic a structured covariance.
typedef struct {
    double structure_scale;  // Scaling factor for the structured noise component.
} StructuredPosteriorData;

// Create a structured posterior object with the specified structure_scale.
Posterior* create_structured_posterior(double structure_scale);

#endif // POSTERIOR_STRUCTURED_H

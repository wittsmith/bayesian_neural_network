#ifndef PRIOR_MIXTURE_H
#define PRIOR_MIXTURE_H

#include "prior.h"

// Create a mixture-of-Gaussians prior with specified parameters.
// For simplicity, this example uses a two-component mixture.
// Inputs: mu1, sigma1 (for first Gaussian), mu2, sigma2 (for second Gaussian),
// and a mixing coefficient lambda (0 <= lambda <= 1) for the first component.
Prior* create_mixture_prior(double mu1, double sigma1, double mu2, double sigma2, double lambda);

#endif // PRIOR_MIXTURE_H

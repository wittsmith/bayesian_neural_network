#ifndef PRIOR_LAPLACE_H
#define PRIOR_LAPLACE_H

#include "prior.h"

// Create a Laplace prior with specified location and scale.
Prior* create_laplace_prior(double location, double scale);

#endif // PRIOR_LAPLACE_H

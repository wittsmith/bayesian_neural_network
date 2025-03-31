#ifndef STOCHASTIC_ACTIVATION_H
#define STOCHASTIC_ACTIVATION_H

#include "../utils/math_utils.h"  // For the Matrix type
#include "../priors/prior.h"      // For the common Prior interface
#include "../posteriors/posterior.h"  // For the common Posterior interface
#include "../config/config.h"

// Structure representing a stochastic activation function (e.g., stochastic PReLU).
// In this example, we implement a stochastic PReLU where the negative slope is random.
// In stochastic_activation.h, add these fields:
typedef struct {
    double alpha_mean;    // Mean value of the negative slope parameter.
    double alpha_logvar;  // Log variance of the negative slope parameter.
    Prior *prior;         // Pointer to Prior interface.
    Posterior *posterior; // Pointer to Posterior interface.
    // --- Added for backward pass ---
    Matrix *cached_input; // Stores the input passed to forward().
    double alpha_sample;  // The value of alpha used during the forward pass.
    double d_alpha_mean;  // Accumulator for the gradient w.r.t. alpha_mean.
    // Optionally, you might add a d_alpha_logvar if you wish to update log-variance.
} StochasticActivation;


// Create a new stochastic activation (stochastic PReLU) with given parameters.
StochasticActivation* create_stochastic_activation(double alpha_mean, double alpha_logvar);

// Free the memory allocated for a stochastic activation function.
void free_stochastic_activation(StochasticActivation *act);

// Forward pass for the stochastic activation function applied element-wise to a matrix.
// For each element x:
//    if x >= 0, output = x;
//    if x < 0, output = alpha * x, where alpha is sampled using the reparameterization trick if stochastic is nonzero,
//    otherwise, alpha = alpha_mean.
// If a Posterior object is provided, its sample() function is used for sampling.
Matrix* stochastic_activation_forward(StochasticActivation *act, const Matrix *input, int stochastic);

// Compute the KL divergence for the stochastic activation parameters using the Prior interface.
// If a Prior is set, it uses its compute_kl() function; otherwise, it falls back to a default Gaussian KL divergence.
double stochastic_activation_kl(StochasticActivation *act);

Matrix* stochastic_activation_backward(void *layer, const Matrix *grad_output, const Config *cfg);

#endif // STOCHASTIC_ACTIVATION_H

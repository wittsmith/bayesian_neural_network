#ifndef BAYESIAN_LINEAR_H
#define BAYESIAN_LINEAR_H

#include "../utils/math_utils.h"   // For Matrix definition and operations.
#include "../bnn_util.h"           // For sample_gaussian and KL divergence helpers.
#include "../priors/prior.h"       // For the common Prior interface.
#include "../posteriors/posterior.h" // For the common Posterior interface

// Structure representing a Bayesian fully-connected (linear) layer.
// Weights and biases are represented by learned means and log-variances.
typedef struct {
    int input_dim;
    int output_dim;
    Matrix *W_mean;    // Weight means: dimensions (output_dim x input_dim)
    Matrix *W_logvar;  // Weight log-variances: same dimensions as W_mean
    double *b_mean;    // Bias means: array of length output_dim
    double *b_logvar;  // Bias log-variances: array of length output_dim
    // Pointer to a Prior structure for KL divergence computation.
    Prior *prior;
    // Pointer to a Posterior structure for weight sampling.
    Posterior *posterior;
} BayesianLinear;

// Create a Bayesian linear layer with given input and output dimensions.
// Initializes weights and biases with small random values for the means and a constant for log-variances.
// The Prior and Posterior pointers are initialized to NULL and should be set externally based on configuration.
BayesianLinear* create_bayesian_linear(int input_dim, int output_dim);

// Free the memory allocated for a Bayesian linear layer.
void free_bayesian_linear(BayesianLinear *layer);

// Forward pass for the Bayesian linear layer.
// If 'stochastic' is nonzero, sample weights and biases using the reparameterization trick;
// if a Posterior object is provided, use its sample() function; otherwise, use sample_gaussian() directly.
// 'input' is a Matrix of shape (num_samples x input_dim).
// Returns a new Matrix of shape (num_samples x output_dim).
Matrix* bayesian_linear_forward(BayesianLinear *layer, const Matrix *input, int stochastic);

// Compute the total KL divergence for this layer using the Prior interface.
// For each weight and bias, if a Prior is set, use its compute_kl() function; otherwise, fall back to a default Gaussian KL divergence.
double bayesian_linear_kl(BayesianLinear *layer, double default_variance);

#endif // BAYESIAN_LINEAR_H

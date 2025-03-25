#ifndef BAYESIAN_CONV_H
#define BAYESIAN_CONV_H

#include "../utils/utils.h"
#include "../bnn_util.h"
#include "../utils/random_utils.h"
#include "../priors/prior.h"       // For the Prior interface
#include "../posteriors/posterior.h" // For the Posterior interface
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Minimal Tensor structure to represent a 3D image-like input.
typedef struct {
    int channels;
    int height;
    int width;
    double *data;  // Stored in order: channel, row, column.
} Tensor;

// Helper functions for Tensor management.
Tensor* create_tensor(int channels, int height, int width);
void free_tensor(Tensor *t);

// Structure representing a Bayesian Convolutional Layer.
typedef struct {
    int input_channels;
    int output_channels;
    int kernel_height;
    int kernel_width;
    // Weight parameters: stored as a flat array of size:
    // output_channels * input_channels * kernel_height * kernel_width.
    double *W_mean;
    double *W_logvar;
    // Bias parameters: arrays of length output_channels.
    double *b_mean;
    double *b_logvar;
    // Pointer to a Prior structure for KL divergence computation.
    Prior *prior;
    // NEW: Pointer to a Posterior structure for sampling the weights and biases.
    Posterior *posterior;
} BayesianConv;

// Create a Bayesian Convolutional layer with specified dimensions.
BayesianConv* create_bayesian_conv(int input_channels, int output_channels, int kernel_height, int kernel_width);

// Free memory allocated for a Bayesian Convolutional layer.
void free_bayesian_conv(BayesianConv *layer);

// Forward pass for the Bayesian Convolutional layer.
// 'input' is a Tensor of shape (input_channels, height, width).
// If 'stochastic' is nonzero, weights and biases are sampled via the reparameterization trick.
// If a Posterior object is provided, its sample() function is used for sampling.
// Returns a new Tensor representing the output (shape: (output_channels, out_height, out_width))
// with out_height = input->height - kernel_height + 1, out_width = input->width - kernel_width + 1.
Tensor* bayesian_conv_forward(BayesianConv *layer, const Tensor *input, int stochastic);

// Compute the total KL divergence for this convolutional layer using the Prior interface.
// For each weight and bias, if a Prior is assigned, it uses layer->prior->compute_kl();
// otherwise, it falls back to a default Gaussian prior (variance = 1.0).
double bayesian_conv_kl(BayesianConv *layer);

#endif // BAYESIAN_CONV_H

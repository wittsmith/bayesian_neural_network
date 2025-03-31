#include "stochastic_activation.h"
#include "../bnn_util.h"   // For sample_gaussian() and kl_divergence_single()
#include "../utils/utils.h"  // For handle_error()
#include <stdlib.h>
#include <math.h>
#include "stochastic_activation.h"
#include "../bnn_util.h"   // For sample_gaussian() and kl_divergence_single()
#include "../utils/utils.h"  // For handle_error()
#include "../config/config.h"
#include <stdlib.h>
#include <math.h>

// Backward pass for the stochastic activation layer.
// It assumes that act->cached_input and act->alpha_sample have been set during the forward pass.
Matrix* stochastic_activation_backward(void *layer, const Matrix *grad_output, const Config *cfg) {
    StochasticActivation *act = (StochasticActivation*) layer;
    if (!act || !act->cached_input || !grad_output) {
        handle_error("Invalid input to stochastic_activation_backward.");
    }
    
    // Create a matrix to hold the gradient with respect to the input.
    Matrix *grad_input = create_matrix(act->cached_input->rows, act->cached_input->cols);
    
    // Initialize the gradient for the alpha parameter.
    double grad_alpha = 0.0;
    
    int total_elements = act->cached_input->rows * act->cached_input->cols;
    for (int i = 0; i < total_elements; i++) {
        double x = act->cached_input->data[i];
        double grad_out = grad_output->data[i];
        // For x >= 0, derivative is 1; for x < 0, derivative is alpha (sampled value).
        if (x >= 0) {
            grad_input->data[i] = grad_out * 1.0;
        } else {
            grad_input->data[i] = grad_out * act->alpha_sample;
            // The derivative of (alpha * x) with respect to alpha is x.
            grad_alpha += grad_out * x;
        }
    }
    
    // Incorporate the KL divergence gradient contribution for the parameter.
    // Here we follow the convention used in bayesian_linear_backward:
    // gradient contribution += kl_weight * (parameter value)
    grad_alpha += cfg->kl_weight * act->alpha_mean;
    
    // Store the computed gradient in the activation layer structure.
    // (Later, your optimizer can use act->d_alpha_mean to update alpha_mean.)
    act->d_alpha_mean = grad_alpha;
    
    // Optionally, if you wish to compute gradients with respect to alpha_logvar,
    // you can add that here similarly.
    
    // Free the cached input as it is no longer needed.
    free_matrix(act->cached_input);
    act->cached_input = NULL;
    
    return grad_input;
}


// Create a new stochastic activation (stochastic PReLU).
StochasticActivation* create_stochastic_activation(double alpha_mean, double alpha_logvar) {
    StochasticActivation *act = (StochasticActivation*)malloc(sizeof(StochasticActivation));
    if (!act) {
        handle_error("Failed to allocate StochasticActivation.");
    }
    act->alpha_mean = alpha_mean;
    act->alpha_logvar = alpha_logvar;
    // Initialize the Prior and Posterior pointers to NULL; they should be set later based on configuration.
    act->prior = NULL;
    act->posterior = NULL;
    return act;
}

// Free the stochastic activation.
void free_stochastic_activation(StochasticActivation *act) {
    if (act) {
        free(act);
    }
}

// Forward pass for stochastic activation.
// Applies a stochastic PReLU element-wise to the input matrix.
// For each element x, if x >= 0, output is x; if x < 0, output is alpha * x,
// where alpha is sampled from N(alpha_mean, exp(alpha_logvar)) if stochastic is true,
// or equals alpha_mean otherwise. If a Posterior object is provided, its sample() function is used.
Matrix* stochastic_activation_forward(StochasticActivation *act, const Matrix *input, int stochastic) {
    if (!act || !input) {
        handle_error("Invalid input to stochastic_activation_forward.");
    }
    
    // Cache the input for the backward pass.
    if (act->cached_input) {
        free_matrix(act->cached_input);
    }
    act->cached_input = copy_matrix(input);
    
    Matrix *output = create_matrix(input->rows, input->cols);
    double alpha;
    if (stochastic) {
        if (act->posterior != NULL) {
            alpha = act->posterior->sample(act->posterior, act->alpha_mean, act->alpha_logvar);
        } else {
            alpha = sample_gaussian(act->alpha_mean, act->alpha_logvar);
        }
    } else {
        alpha = act->alpha_mean;
    }
    // Save the sampled alpha for use in the backward pass.
    act->alpha_sample = alpha;
    
    int total_elements = input->rows * input->cols;
    for (int i = 0; i < total_elements; i++) {
        double x = input->data[i];
        output->data[i] = (x >= 0) ? x : alpha * x;
    }
    return output;
}


// Compute the KL divergence for the stochastic activation parameters.
// If a Prior is set, use its compute_kl() function; otherwise, fall back to a default Gaussian KL divergence with variance 1.0.
double stochastic_activation_kl(StochasticActivation *act) {
    if (!act) {
        handle_error("Invalid StochasticActivation in KL computation.");
    }
    if (act->prior == NULL) {
        double default_variance = 1.0;
        return kl_divergence_single(act->alpha_mean, act->alpha_logvar, default_variance);
    } else {
        return act->prior->compute_kl(act->prior, act->alpha_mean, act->alpha_logvar);
    }
}

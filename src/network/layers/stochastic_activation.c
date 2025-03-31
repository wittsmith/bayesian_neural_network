#include "stochastic_activation.h"
#include "../bnn_util.h"   // For sample_gaussian() and kl_divergence_single()
#include "../utils/utils.h"  // For handle_error()
#include "../config/config.h"
#include "../priors/prior_laplace.h"
#include "../priors/prior_mixture.h"
#include "../posteriors/posterior_flipout.h"
#include "../posteriors/posterior_structured.h"
#include <stdlib.h>
#include <math.h>

// Helper function to clip gradients
static double clip_gradient(double grad, double clip_value) {
    if (grad > clip_value) return clip_value;
    if (grad < -clip_value) return -clip_value;
    return grad;
}

// Backward pass for the stochastic activation layer.
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
        
        // Apply noise injection if configured
        if (cfg->noise_injection > 0.0) {
            grad_out += sample_gaussian(0.0, cfg->noise_injection);
        }
        
        // For x >= 0, derivative is 1; for x < 0, derivative is alpha (sampled value).
        if (x >= 0) {
            grad_input->data[i] = grad_out * 1.0;
        } else {
            grad_input->data[i] = grad_out * act->alpha_sample;
            // The derivative of (alpha * x) with respect to alpha is x.
            grad_alpha += grad_out * x;
        }
    }
    
    // Apply gradient clipping if configured
    if (cfg->grad_clip > 0.0) {
        grad_alpha = clip_gradient(grad_alpha, cfg->grad_clip);
    }
    
    // Incorporate the KL divergence gradient contribution for the parameter.
    // Use the configured prior variance and KL weight
    double kl_contrib = cfg->kl_weight * act->alpha_mean;
    if (cfg->kl_annealing) {
        // If KL annealing is enabled, scale the KL contribution
        kl_contrib *= (1.0 - exp(-cfg->kl_weight * cfg->num_epochs));
    }
    grad_alpha += kl_contrib;
    
    // Store the computed gradient in the activation layer structure.
    act->d_alpha_mean = grad_alpha;
    
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
    act->prior_variance = 1.0;  // Default prior variance
    act->prior = NULL;
    act->posterior = NULL;
    act->cached_input = NULL;
    act->alpha_sample = 0.0;
    act->d_alpha_mean = 0.0;
    return act;
}

// Forward pass for stochastic activation.
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
            // Use the configured posterior method for sampling
            alpha = act->posterior->sample(act->posterior, act->alpha_mean, act->alpha_logvar);
        } else {
            // Fall back to standard reparameterization trick
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
double stochastic_activation_kl(StochasticActivation *act) {
    if (!act) {
        handle_error("Invalid StochasticActivation in KL computation.");
    }
    if (act->prior == NULL) {
        // Use the configured prior variance instead of hardcoded 1.0
        return kl_divergence_single(act->alpha_mean, act->alpha_logvar, act->prior_variance);
    } else {
        return act->prior->compute_kl(act->prior, act->alpha_mean, act->alpha_logvar);
    }
}

// Free the stochastic activation.
void free_stochastic_activation(StochasticActivation *act) {
    if (act) {
        if (act->cached_input) {
            free_matrix(act->cached_input);
        }
        // Free the prior and posterior if they exist
        if (act->prior) {
            if (act->prior->data) {
                free(act->prior->data);
            }
            free(act->prior);
        }
        if (act->posterior) {
            if (act->posterior->data) {
                free(act->posterior->data);
            }
            free(act->posterior);
        }
        free(act);
    }
}

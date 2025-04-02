#include "adam_optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Initialize Adam state for a layer
AdamState* init_adam_state(int size) {
    AdamState* state = (AdamState*)malloc(sizeof(AdamState));
    if (!state) return NULL;
    
    state->size = size;
    state->t = 0;
    
    // Allocate memory for moment vectors
    state->m = (double*)calloc(size, sizeof(double));
    state->v = (double*)calloc(size, sizeof(double));
    
    if (!state->m || !state->v) {
        free_adam_state(state);
        return NULL;
    }
    
    return state;
}

// Free Adam state
void free_adam_state(AdamState* state) {
    if (!state) return;
    
    if (state->m) free(state->m);
    if (state->v) free(state->v);
    free(state);
}

// Helper function to update moments and parameters for a vector of parameters
void update_moments_and_params(
    double* params,
    double* grads,
    double* m,
    double* v,
    int size,
    const Config* cfg,
    int t
) {
    double beta1 = cfg->adam_beta1;
    double beta2 = cfg->adam_beta2;
    double epsilon = cfg->adam_epsilon;
    double lr = cfg->learning_rate;
    
    // Compute bias correction terms
    double m_hat, v_hat;
    double beta1_t = pow(beta1, t);
    double beta2_t = pow(beta2, t);
    
    for (int i = 0; i < size; i++) {
        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        
        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        
        // Compute bias-corrected first moment estimate
        m_hat = m[i] / (1.0 - beta1_t);
        
        // Compute bias-corrected second raw moment estimate
        v_hat = v[i] / (1.0 - beta2_t);
        
        // Update parameters
        params[i] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

// Update parameters for BayesianLinear layer using Adam
void adam_update_bayesian_linear(BayesianLinear *layer, AdamState *state, const Config *cfg) {
    if (!layer || !state || !cfg) return;
    
    // Increment time step
    state->t++;
    
    int total_weights = layer->output_dim * layer->input_dim;
    
    // Update weights
    update_moments_and_params(
        layer->W_mean->data,
        layer->dW_mean->data,
        state->m,
        state->v,
        total_weights,
        cfg,
        state->t
    );
    
    // Update biases
    update_moments_and_params(
        layer->b_mean,
        layer->db_mean,
        state->m + total_weights,
        state->v + total_weights,
        layer->output_dim,
        cfg,
        state->t
    );
    
    // Reset gradients
    memset(layer->dW_mean->data, 0, total_weights * sizeof(double));
    memset(layer->db_mean, 0, layer->output_dim * sizeof(double));
}

// Update parameters for StochasticActivation layer using Adam
void adam_update_stochastic_activation(StochasticActivation *layer, AdamState *state, const Config *cfg) {
    if (!layer || !state || !cfg) return;
    
    // Increment time step
    state->t++;
    
    // Update alpha parameter
    update_moments_and_params(
        &layer->alpha_mean,
        &layer->d_alpha_mean,
        state->m,
        state->v,
        1,  // Only one parameter to update
        cfg,
        state->t
    );
    
    // Reset gradient
    layer->d_alpha_mean = 0.0;
} 
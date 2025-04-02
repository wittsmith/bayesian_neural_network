#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "../network/layers/bayesian_linear.h"
#include "../network/layers/stochastic_activation.h"
#include "../config/config.h"

// Adam optimizer state structure
typedef struct {
    double *m;      // First moment vector
    double *v;      // Second moment vector
    int size;       // Size of the parameter vector
    int t;          // Time step
} AdamState;

// Initialize Adam state for a layer
AdamState* init_adam_state(int size);

// Free Adam state
void free_adam_state(AdamState* state);

// Update moments and parameters using Adam optimizer
void update_moments_and_params(
    double* params,
    double* grads,
    double* m,
    double* v,
    int size,
    const Config* cfg,
    int t
);

// Update parameters using Adam optimizer
void adam_update_bayesian_linear(BayesianLinear *layer, AdamState *state, const Config *cfg);
void adam_update_stochastic_activation(StochasticActivation *layer, AdamState *state, const Config *cfg);

#endif // ADAM_OPTIMIZER_H 
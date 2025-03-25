#include "stochastic_activation.h"
#include "../bnn_util.h"   // For sample_gaussian() and kl_divergence_single()
#include "../utils/utils.h"  // For handle_error()
#include <stdlib.h>
#include <math.h>

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

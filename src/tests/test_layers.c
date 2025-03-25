#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../config/config.h"
#include "layers/bayesian_linear.h"
#include "layers/bayesian_conv.h"
#include "layers/dropout_layer.h"
#include "layers/stochastic_activation.h"
#include "priors/prior_laplace.h"
#include "posteriors/posterior_flipout.h"
#include "posteriors/posterior_structured.h"
#include "../utils/utils.h"

int main() {
    // Initialize configuration with default values.
    Config cfg;
    init_config(&cfg);
    
    // Override configuration for testing.
    cfg.prior_type = 1;          // Use Laplace prior.
    cfg.posterior_method = 2;    // Use Flipout posterior.
    
    // --- Test BayesianLinear Layer ---
    int input_dim = 50, output_dim = 20;
    BayesianLinear *bl = create_bayesian_linear(input_dim, output_dim);
    // Assign Prior and Posterior based on test config.
    bl->prior = create_laplace_prior(0.0, cfg.prior_variance);
    bl->posterior = create_flipout_posterior();
    
    // Verify that Prior and Posterior pointers are set.
    assert(bl->prior != NULL);
    assert(bl->posterior != NULL);
    printf("BayesianLinear layer created with Prior and Posterior assigned.\n");
    free_bayesian_linear(bl);
    
    // --- Test BayesianConv Layer ---
    int in_channels = 3, out_channels = 8, kernel_h = 3, kernel_w = 3;
    BayesianConv *bc = create_bayesian_conv(in_channels, out_channels, kernel_h, kernel_w);
    bc->prior = create_laplace_prior(0.0, cfg.prior_variance);
    bc->posterior = create_flipout_posterior();
    
    // Verify that Prior and Posterior pointers are set.
    assert(bc->prior != NULL);
    assert(bc->posterior != NULL);
    printf("BayesianConv layer created with Prior and Posterior assigned.\n");
    free_bayesian_conv(bc);
    
    // --- Test Dropout Layer ---
    DropoutLayer *dl = create_dropout_layer(DROPOUT_MC, cfg.dropout_prob, 0.0);
    // Dropout layers typically do not use Prior/Posterior.
    assert(dl != NULL);
    printf("Dropout layer created successfully.\n");
    free_dropout_layer(dl);
    
    // --- Test StochasticActivation Layer ---
    double alpha_mean = 0.25, alpha_logvar = -5.0;
    StochasticActivation *sa = create_stochastic_activation(alpha_mean, alpha_logvar);
    sa->prior = create_laplace_prior(0.0, cfg.prior_variance);
    sa->posterior = create_flipout_posterior();
    
    // Verify that Prior and Posterior pointers are set.
    assert(sa->prior != NULL);
    assert(sa->posterior != NULL);
    printf("StochasticActivation layer created with Prior and Posterior assigned.\n");
    free_stochastic_activation(sa);
    
    printf("All layer creation tests passed successfully.\n");
    return 0;
}

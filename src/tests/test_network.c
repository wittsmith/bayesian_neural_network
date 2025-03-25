#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../config/config.h"
#include "../network/network.h"
#include "../utils/math_utils.h"
#include "../utils/utils.h"
#include <string.h>

int main(void) {
    // Initialize configuration with defaults.
    Config cfg;
    init_config(&cfg);
    
    // Override configuration for testing.
    // Set number of layers, neuron counts, and layer types.
    // For example: 3 layers with "128,64,10" neurons and corresponding types.
    cfg.num_layers = 3;
    strncpy(cfg.neurons_per_layer, "128,64,10", sizeof(cfg.neurons_per_layer) - 1);
    cfg.neurons_per_layer[sizeof(cfg.neurons_per_layer) - 1] = '\0';
    // Specify layer types: valid values: "linear", "conv", "dropout", "stochastic"
    strncpy(cfg.layer_types, "linear,dropout,stochastic", sizeof(cfg.layer_types) - 1);
    cfg.layer_types[sizeof(cfg.layer_types) - 1] = '\0';
    
    // Set Prior and Posterior configuration.
    cfg.prior_type = 1;          // Use Laplace prior.
    cfg.prior_variance = 1.0;
    cfg.posterior_method = 2;    // Use Flipout posterior.
    
    // Set dropout probability (used by dropout layers).
    cfg.dropout_prob = 0.5;
    
    // Print configuration info (for debugging).
    printf("Creating network with %d layers\n", cfg.num_layers);
    printf("Neurons per layer: %s\n", cfg.neurons_per_layer);
    printf("Layer types: %s\n", cfg.layer_types);
    printf("Prior type: %d, Posterior method: %d\n", cfg.prior_type, cfg.posterior_method);
    
    // Create the network.
    Network *net = create_network(&cfg);
    assert(net != NULL);
    printf("Network created with %d layers.\n", net->num_layers);
    
    // Create a synthetic input matrix.
    // Assume input dimension is 100 (as used in create_network).
    int batch_size = 10;
    int input_dim = 100;
    Matrix *input = create_matrix(batch_size, input_dim);
    // Fill the input matrix with a constant value (e.g., 1.0).
    for (int i = 0; i < batch_size * input_dim; i++) {
        input->data[i] = 1.0;
    }
    
    // Run the forward pass in stochastic mode.
    Matrix *output = network_forward(net, input, 1);
    printf("Forward pass completed.\n");
    printf("Output dimensions: %d x %d\n", output->rows, output->cols);
    
    // Compute the total KL divergence for the network.
    double total_kl = network_total_kl(net);
    printf("Total KL divergence: %f\n", total_kl);
    
    // Clean up allocated resources.
    free_matrix(output);
    free_matrix(input);
    free_network(net);
    
    printf("Network test completed successfully.\n");
    return 0;
}

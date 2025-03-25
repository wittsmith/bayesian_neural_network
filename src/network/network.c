#include "network.h"
#include "../config/config.h"
#include "../utils/utils.h"
#include <stdlib.h>
#include <string.h>

// Include layer headers.
#include "../layers/bayesian_linear.h"
#include "../layers/bayesian_conv.h"
#include "../layers/dropout_layer.h"
#include "../layers/stochastic_activation.h"
// (Include others as needed, e.g., noise injection)

// Include Prior and Posterior creation functions.
#include "../priors/prior_laplace.h"
#include "../priors/prior_mixture.h"
#include "../posteriors/posterior_flipout.h"
#include "../posteriors/posterior_structured.h"

// ==================
// Helper Functions for Layer Wrappers
// ==================

// --- BayesianLinear ---
static double linear_kl_wrapper(void *layer_ptr) {
    return bayesian_linear_kl((BayesianLinear*)layer_ptr);
}

static Layer* create_linear_layer(BayesianLinear *bl) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) {
        handle_error("Failed to allocate Layer for BayesianLinear.");
    }
    l->layer = (void*)bl;
    l->forward = (Matrix* (*)(void*, const Matrix*, int)) bayesian_linear_forward;
    l->kl = linear_kl_wrapper;
    l->free_layer = (void (*)(void*)) free_bayesian_linear;
    return l;
}

// --- BayesianConv ---
static double conv_kl_wrapper(void *layer_ptr) {
    return bayesian_conv_kl((BayesianConv*)layer_ptr);
}

static Layer* create_conv_layer(BayesianConv *bc) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) {
        handle_error("Failed to allocate Layer for BayesianConv.");
    }
    l->layer = (void*)bc;
    l->forward = (Matrix* (*)(void*, const Matrix*, int)) bayesian_conv_forward;
    l->kl = conv_kl_wrapper;
    l->free_layer = (void (*)(void*)) free_bayesian_conv;
    return l;
}

// --- DropoutLayer ---
static double dropout_kl_wrapper(void *layer_ptr) {
    // Dropout layers typically have no KL divergence if no learnable parameters.
    return 0.0;
}

static Layer* create_dropout_layer_wrapper(DropoutLayer *dl) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) {
        handle_error("Failed to allocate Layer for DropoutLayer.");
    }
    l->layer = (void*)dl;
    l->forward = (Matrix* (*)(void*, const Matrix*, int)) dropout_forward;
    l->kl = dropout_kl_wrapper;
    l->free_layer = (void (*)(void*)) free_dropout_layer;
    return l;
}

// --- StochasticActivation ---
static double stochastic_act_kl_wrapper(void *layer_ptr) {
    return stochastic_activation_kl((StochasticActivation*)layer_ptr);
}

static Layer* create_stochastic_act_layer_wrapper(StochasticActivation *sa) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) {
        handle_error("Failed to allocate Layer for StochasticActivation.");
    }
    l->layer = (void*)sa;
    l->forward = (Matrix* (*)(void*, const Matrix*, int)) stochastic_activation_forward;
    l->kl = stochastic_act_kl_wrapper;
    l->free_layer = (void (*)(void*)) free_stochastic_activation;
    return l;
}

// ==================
// create_network: Constructs the network from config settings.
// ==================
Network* create_network(const Config *cfg) {
    if (!cfg) {
        handle_error("Config is NULL in create_network.");
    }
    
    // Allocate the network structure.
    Network *net = (Network*)malloc(sizeof(Network));
    if (!net) {
        handle_error("Failed to allocate Network structure.");
    }
    
    // Parse neurons_per_layer to get sizes (for linear and conv layers, as applicable).
    int layer_sizes[10];
    int num_sizes = 0;
    char neurons_copy[256];
    strncpy(neurons_copy, cfg->neurons_per_layer, sizeof(neurons_copy));
    char *token = strtok(neurons_copy, ",");
    while (token != NULL && num_sizes < 10) {
        layer_sizes[num_sizes++] = atoi(token);
        token = strtok(NULL, ",");
    }
    
    // Parse layer_types to determine the type for each layer.
    // For simplicity, assume at most 10 layers.
    char *layer_types[10];
    int num_types = 0;
    char types_copy[256];
    strncpy(types_copy, cfg->layer_types, sizeof(types_copy));
    token = strtok(types_copy, ",");
    while (token != NULL && num_types < 10) {
        // Duplicate token string to store in array.
        layer_types[num_types] = strdup(token);
        num_types++;
        token = strtok(NULL, ",");
    }
    
    // Use the smaller of num_sizes and num_types as the number of layers.
    int num_layers = (num_sizes < num_types) ? num_sizes : num_types;
    net->num_layers = num_layers;
    net->layers = (Layer**)malloc(sizeof(Layer*) * num_layers);
    if (!net->layers) {
        handle_error("Failed to allocate layers array in create_network.");
    }
    
    // For demonstration, define a starting input dimension.
    int input_dim = 100; // Placeholder; should come from data dimensions.
    for (int i = 0; i < num_layers; i++) {
        // Decide layer type from layer_types[i] (convert to lower-case if needed).
        // For simplicity, we use strcmp; in a robust system, add error checking.
        if (strcmp(layer_types[i], "linear") == 0) {
            // Create BayesianLinear layer.
            int output_dim = layer_sizes[i];
            BayesianLinear *bl = create_bayesian_linear(input_dim, output_dim);
            
            // Set Prior based on cfg->prior_type.
            if (cfg->prior_type == 1) {
                bl->prior = create_laplace_prior(0.0, cfg->prior_variance);
            } else if (cfg->prior_type == 2) {
                bl->prior = create_mixture_prior(0.0, 1.0, 0.0, 1.0, 0.5);
            } else {
                bl->prior = NULL; // Default Gaussian.
            }
            
            // Set Posterior based on cfg->posterior_method.
            if (cfg->posterior_method == 2) {
                bl->posterior = create_flipout_posterior();
            } else if (cfg->posterior_method == 1) {
                bl->posterior = create_structured_posterior(1.0);
            } else {
                bl->posterior = NULL; // Default mean-field.
            }
            
            net->layers[i] = create_linear_layer(bl);
            input_dim = output_dim; // Update input dimension for next layer.
        } else if (strcmp(layer_types[i], "conv") == 0) {
            // Create BayesianConv layer.
            // For demonstration, use fixed parameters for kernel size.
            int output_channels = layer_sizes[i];
            int kernel_height = 3, kernel_width = 3;
            BayesianConv *bc = create_bayesian_conv(input_dim, output_channels, kernel_height, kernel_width);
            
            // Set Prior.
            if (cfg->prior_type == 1) {
                bc->prior = create_laplace_prior(0.0, cfg->prior_variance);
            } else if (cfg->prior_type == 2) {
                bc->prior = create_mixture_prior(0.0, 1.0, 0.0, 1.0, 0.5);
            } else {
                bc->prior = NULL;
            }
            // Set Posterior.
            if (cfg->posterior_method == 2) {
                bc->posterior = create_flipout_posterior();
            } else if (cfg->posterior_method == 1) {
                bc->posterior = create_structured_posterior(1.0);
            } else {
                bc->posterior = NULL;
            }
            
            net->layers[i] = create_conv_layer(bc);
            // For convolution, input dimension for the next layer is not simply output_channels.
            // Here, we update input_dim as a placeholder.
            input_dim = output_channels; 
        } else if (strcmp(layer_types[i], "dropout") == 0) {
            // Create a Dropout layer.
            DropoutLayer *dl = create_dropout_layer(DROPOUT_MC, cfg->dropout_prob, 0.0);
            net->layers[i] = create_dropout_layer_wrapper(dl);
            // For dropout, output dimension remains the same as input.
        } else if (strcmp(layer_types[i], "stochastic") == 0) {
            // Create a StochasticActivation layer.
            // For demonstration, set the activation parameter arbitrarily.
            double alpha_mean = 0.25, alpha_logvar = -5.0;
            StochasticActivation *sa = create_stochastic_activation(alpha_mean, alpha_logvar);
            // Set Prior if desired.
            if (cfg->prior_type == 1) {
                sa->prior = create_laplace_prior(0.0, cfg->prior_variance);
            } else if (cfg->prior_type == 2) {
                sa->prior = create_mixture_prior(0.0, 1.0, 0.0, 1.0, 0.5);
            } else {
                sa->prior = NULL;
            }
            // Set Posterior for stochastic activation.
            if (cfg->posterior_method == 2) {
                sa->posterior = create_flipout_posterior();
            } else if (cfg->posterior_method == 1) {
                sa->posterior = create_structured_posterior(1.0);
            } else {
                sa->posterior = NULL;
            }
            net->layers[i] = create_stochastic_act_layer_wrapper(sa);
            // Output dimension is the same as input dimension.
        } else {
            // If the type is unrecognized, default to a BayesianLinear layer.
            int output_dim = layer_sizes[i];
            BayesianLinear *bl = create_bayesian_linear(input_dim, output_dim);
            bl->prior = NULL;
            bl->posterior = NULL;
            net->layers[i] = create_linear_layer(bl);
            input_dim = output_dim;
        }
    }
    
    // Free the duplicated layer type strings.
    for (int i = 0; i < num_types; i++) {
        free(layer_types[i]);
    }
    
    return net;
}

// ==================
// Forward pass: Propagate input through the network.
// ==================
Matrix* network_forward(Network *net, const Matrix *input, int stochastic) {
    if (!net || !input) {
        handle_error("Null network or input in network_forward.");
    }
    Matrix *current = (Matrix*)input;  // Do not free the original input.
    for (int i = 0; i < net->num_layers; i++) {
        Matrix *next = net->layers[i]->forward(net->layers[i]->layer, current, stochastic);
        if (i > 0) {  // Free intermediate outputs.
            free_matrix(current);
        }
        current = next;
    }
    return current;
}

// ==================
// Total KL divergence: Sum KL contributions from each Bayesian layer.
// ==================
double network_total_kl(Network *net) {
    if (!net) {
        handle_error("Null network in network_total_kl.");
    }
    double total_kl = 0.0;
    for (int i = 0; i < net->num_layers; i++) {
        total_kl += net->layers[i]->kl(net->layers[i]->layer);
    }
    return total_kl;
}

// ==================
// Free the network and all its layers.
// ==================
void free_network(Network *net) {
    if (net) {
        for (int i = 0; i < net->num_layers; i++) {
            if (net->layers[i]) {
                net->layers[i]->free_layer(net->layers[i]->layer);
                free(net->layers[i]);
            }
        }
        free(net->layers);
        free(net);
    }
}

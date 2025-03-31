#include "network.h"
#include "../config/config.h"
#include "../utils/utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include layer headers.
#include "layers/bayesian_linear.h"
#include "layers/bayesian_conv.h"
#include "layers/dropout_layer.h"
#include "layers/stochastic_activation.h"

// Include Prior and Posterior creation functions.
#include "priors/prior_laplace.h"
#include "priors/prior_mixture.h"
#include "posteriors/posterior_flipout.h"
#include "posteriors/posterior_structured.h"

#include "../optimizer/optimizer.h"

// ==================
// Helper Functions for Layer Wrappers
// ------------------ (unchanged)
static double linear_kl_wrapper(void *layer_ptr) {
    return bayesian_linear_kl((BayesianLinear*)layer_ptr, 1.0);
}
static Layer* create_linear_layer(BayesianLinear *bl, const Config *cfg) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    l->layer = (void*)bl;
    l->type = LAYER_BAYESIAN_LINEAR;
    l->forward = (Matrix* (*)(void*, const Matrix*, int)) bayesian_linear_forward;
    l->backward = (Matrix* (*)(void*, const Matrix*, const Config*)) bayesian_linear_backward;
    l->kl = linear_kl_wrapper;
    l->free_layer = (void (*)(void*)) free_bayesian_linear;
    return l;
}

static Matrix* conv_forward_wrapper(void *layer, const Matrix *input, int stochastic) {
    // Cast the layer pointer to BayesianConv.
    BayesianConv *bc = (BayesianConv*) layer;
    
    // Infer the Tensor shape from the input Matrix.
    // Assume that the input Matrix has dimensions: [batch_size x flat_dim],
    // where flat_dim = input_channels * (height * width).
    int channels = bc->input_channels;
    int flat = input->cols;
    int spatial = flat / channels;
    int side = (int) sqrt((double) spatial);
    if (side * side * channels != flat) {
        handle_error("Cannot infer Tensor shape for conv layer from Matrix input.");
    }
    
    // Convert the Matrix to a Tensor.
    Tensor *tensor_input = matrix_to_tensor(input, channels, side, side);
    
    // Call the actual convolution forward function.
    Matrix *output = bayesian_conv_forward(bc, tensor_input, stochastic);
    
    // Free the temporary Tensor.
    free_tensor(tensor_input);
    
    return output;
}

static double conv_kl_wrapper(void *layer_ptr) {
    return bayesian_conv_kl((BayesianConv*)layer_ptr);
}
static Layer* create_conv_layer(BayesianConv *bc) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) {
        handle_error("Failed to allocate Layer for BayesianConv.");
    }
    l->layer = (void*)bc;
    l->type = LAYER_BAYESIAN_CONV;
    // Use conv_forward_wrapper to convert Matrix to Tensor before calling bayesian_conv_forward.
    l->forward = conv_forward_wrapper;
    l->kl = conv_kl_wrapper;
    l->free_layer = (void (*)(void*)) free_bayesian_conv;
    return l;
}




static double dropout_kl_wrapper(void *layer_ptr) {
    return 0.0;
}
static Layer* create_dropout_layer_wrapper(DropoutLayer *dl) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) {
        handle_error("Failed to allocate Layer for DropoutLayer.");
    }
    l->layer = (void*)dl;
    l->type = LAYER_DROPOUT;
    l->forward = (Matrix* (*)(void*, const Matrix*, int)) dropout_forward;
    l->backward = (Matrix* (*)(void*, const Matrix*, const Config*)) dropout_backward;
    l->kl = dropout_kl_wrapper;
    l->free_layer = (void (*)(void*)) free_dropout_layer;
    return l;
}

static double stochastic_act_kl_wrapper(void *layer_ptr) {
    return stochastic_activation_kl((StochasticActivation*)layer_ptr);
}


static Layer* create_stochastic_act_layer_wrapper(StochasticActivation *sa) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) {
        handle_error("Failed to allocate Layer for StochasticActivation.");
    }
    l->layer = (void*)sa;
    l->type = LAYER_STOCHASTIC_ACTIVATION;
    l->forward = (Matrix* (*)(void*, const Matrix*, int)) stochastic_activation_forward;
    l->backward = (Matrix* (*)(void*, const Matrix*, const Config*)) stochastic_activation_backward;
    l->kl = stochastic_act_kl_wrapper;
    l->free_layer = (void (*)(void*)) free_stochastic_activation;
    return l;
}




Matrix* network_backward(Network *net, const Matrix *grad_output, const Config *cfg) {
    Matrix *grad = (Matrix*)grad_output;
    //printf("net->num_layers: %d", net->num_layers);
    fflush(stdout);
    for (int i = net->num_layers - 1; i >= 0; i--) {
        //printf("iteration %d counting down ", i);
        if (!net->layers[i]) {
            printf("Layer %d is NULL\n", i);
            exit(1);
        }
        if (!net->layers[i]->backward) {
            printf("Backward function pointer for layer %d is NULL\n", i);
            exit(1);
        }
        if (!net->layers[i]->layer) {
            printf("Internal layer pointer for layer %d is NULL\n", i);
            exit(1);
        }

        fflush(stdout);

        Matrix *new_grad = net->layers[i]->backward(net->layers[i]->layer, grad, cfg);
       // printf("matrix pointer: %p", new_grad);
        fflush(stdout);
        if (i < net->num_layers - 1) {
            free_matrix(grad); // free the old gradient
        }
        grad = new_grad;
    }
    return grad; // gradient w.r.t. the original input, if needed
}


// ------------------
// Helper: Projection Layer
// ------------------
// Create a projection layer (using BayesianLinear) to map from input_dim to target_dim.
// This layer is inserted internally to adjust the dimension but is not counted as a main layer.
static Layer* create_projection_layer(int input_dim, int target_dim, const Config *cfg) {
    BayesianLinear *proj = create_bayesian_linear(input_dim, target_dim);
    proj->prior = NULL;
    proj->posterior = NULL;
    return create_linear_layer(proj, cfg);
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
 //   printf("allocated network structure\n");
    fflush(stdout);
    
    // Parse neurons_per_layer to get sizes.
    int layer_sizes[10];
    int num_sizes = 0;
    char neurons_copy[256];
    strncpy(neurons_copy, cfg->neurons_per_layer, sizeof(neurons_copy));
    char *token = strtok(neurons_copy, ",");
    while (token != NULL && num_sizes < 10) {
        layer_sizes[num_sizes++] = atoi(token);
        token = strtok(NULL, ",");
    }
    
    // Parse layer_types.
    char *layer_types[10];
    int num_types = 0;
    char types_copy[256];
    strncpy(types_copy, cfg->layer_types, sizeof(types_copy));
    token = strtok(types_copy, ",");
    while (token != NULL && num_types < 10) {
        layer_types[num_types] = strdup(token);
        num_types++;
        token = strtok(NULL, ",");
    }
    
    // Use the smaller count as the logical number of layers (as per config).
    int logical_layers = (num_sizes < num_types) ? num_sizes : num_types;
    net->logical_num_layers = logical_layers;
    
    // Allocate an array for the internal (full) layer list.
    // We allow extra space for inserted projection layers.
    Layer **full_layers = (Layer**)malloc(sizeof(Layer*) * (logical_layers + logical_layers)); // worst-case double.
    if (!full_layers) {
        handle_error("Failed to allocate full layers array in create_network.");
    }
   // printf("allocated full layers array\n");
    fflush(stdout);
    int current_index = 0;
    int current_dim = 100;  // Placeholder input dimension.
    
    // Iterate over the logical layers.
    for (int i = 0; i < logical_layers; i++) {
        int target_dim = layer_sizes[i];
        // Get the specified layer type.
        char *type = layer_types[i];
        
        if (strcmp(type, "linear") == 0) {
            // Create a BayesianLinear layer.
            BayesianLinear *bl = create_bayesian_linear(current_dim, target_dim);
            if (cfg->prior_type == 1) {
                bl->prior = create_laplace_prior(0.0, cfg->prior_variance);
            } else if (cfg->prior_type == 2) {
                bl->prior = create_mixture_prior(0.0, 1.0, 0.0, 1.0, 0.5);
            } else {
                bl->prior = NULL;
            }
            if (cfg->posterior_method == 2) {
                bl->posterior = create_flipout_posterior();
            } else if (cfg->posterior_method == 1) {
                bl->posterior = create_structured_posterior(1.0);
            } else {
                bl->posterior = NULL;
            }
            full_layers[current_index++] = create_linear_layer(bl, cfg);
            current_dim = target_dim;
        } else if (strcmp(type, "conv") == 0) {
            // Create a BayesianConv layer.
            int kernel_h = 3, kernel_w = 3;
            BayesianConv *bc = create_bayesian_conv(current_dim, target_dim, kernel_h, kernel_w);
            if (cfg->prior_type == 1) {
                bc->prior = create_laplace_prior(0.0, cfg->prior_variance);
            } else if (cfg->prior_type == 2) {
                bc->prior = create_mixture_prior(0.0, 1.0, 0.0, 1.0, 0.5);
            } else {
                bc->prior = NULL;
            }
            if (cfg->posterior_method == 2) {
                bc->posterior = create_flipout_posterior();
            } else if (cfg->posterior_method == 1) {
                bc->posterior = create_structured_posterior(1.0);
            } else {
                bc->posterior = NULL;
            }
            full_layers[current_index++] = create_conv_layer(bc);
            current_dim = target_dim;
        } else if (strcmp(type, "dropout") == 0) {
            // Create a Dropout layer.
            DropoutLayer *dl = create_dropout_layer(DROPOUT_MC, cfg->dropout_prob, 0.0);
            full_layers[current_index++] = create_dropout_layer_wrapper(dl);
            // Dropout does not change dimensions.
            if (current_dim != target_dim) {
                // Insert an internal projection layer, but do not count it toward logical_layers.
                full_layers[current_index++] = create_projection_layer(current_dim, target_dim, cfg);
                current_dim = target_dim;
            }
        } else if (strcmp(type, "stochastic") == 0) {
            // Create a StochasticActivation layer.
            double alpha_mean = 0.25, alpha_logvar = -5.0;
            StochasticActivation *sa = create_stochastic_activation(alpha_mean, alpha_logvar);
            if (cfg->prior_type == 1) {
                sa->prior = create_laplace_prior(0.0, cfg->prior_variance);
            } else if (cfg->prior_type == 2) {
                sa->prior = create_mixture_prior(0.0, 1.0, 0.0, 1.0, 0.5);
            } else {
                sa->prior = NULL;
            }
            if (cfg->posterior_method == 2) {
                sa->posterior = create_flipout_posterior();
            } else if (cfg->posterior_method == 1) {
                sa->posterior = create_structured_posterior(1.0);
            } else {
                sa->posterior = NULL;
            }
            full_layers[current_index++] = create_stochastic_act_layer_wrapper(sa);
            // Stochastic activation does not change dimensions.
            if (current_dim != target_dim) {
                full_layers[current_index++] = create_projection_layer(current_dim, target_dim, cfg);
                current_dim = target_dim;
            }
        } else {
            // Default to BayesianLinear.
            BayesianLinear *bl = create_bayesian_linear(current_dim, target_dim);
            bl->prior = NULL;
            bl->posterior = NULL;
            full_layers[current_index++] = create_linear_layer(bl, cfg);
            current_dim = target_dim;
        }
    }
    
    // Set the full layer count.
    net->num_layers = current_index;
    // The logical number of layers is as per config.
    // (You could report this separately if needed.)
    net->layers = full_layers;
    
    // Free the temporary layer type strings.
    for (int i = 0; i < num_types; i++) {
        free(layer_types[i]);
    }
    
    return net;
}
// Conv forward wrapper: converts Matrix input to Tensor, calls bayesian_conv_forward, then returns the output Matrix.


// ==================
// Forward pass: Propagate input through the network.
// ==================
Matrix* network_forward(Network *net, const Matrix *input, int stochastic) {
    if (!net || !input) {
        handle_error("Null network or input in network_forward.");
    }
    Matrix *current = (Matrix*)input;  // Do not free the original input.
    
    for (int i = 0; i < net->num_layers; i++) {
        Matrix *next;
        // Check if the layer is a conv layer by comparing the forward pointer.
        if (net->layers[i]->forward == conv_forward_wrapper) {
            // Call the conv forward wrapper directly.
            next = conv_forward_wrapper(net->layers[i]->layer, current, stochastic);
        } else {
            next = net->layers[i]->forward(net->layers[i]->layer, current, stochastic);
        }
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

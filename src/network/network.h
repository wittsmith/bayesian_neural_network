#ifndef NETWORK_H
#define NETWORK_H

#include "../config/config.h"
#include "../utils/math_utils.h"

// Abstract layer interface.
// In network.h
typedef struct Layer {
    void *layer;
    
    // Forward pass
    Matrix* (*forward)(void *layer, const Matrix *input, int stochastic);

    // Backward pass
    //   - grad_output is the gradient of the loss w.r.t. this layer's output
    //   - returns the gradient w.r.t. this layer's input (so it can be passed to the previous layer)
    Matrix* (*backward)(void *layer, const Matrix *grad_output, const Config *cfg);

    // KL divergence
    double (*kl)(void *layer);

    // Free resources
    void (*free_layer)(void *layer);
} Layer;



// Network structure.
typedef struct Network {
    Layer **layers;        // Array of pointers to all (internal) layers (including projection layers).
    int num_layers;        // Total number of layers (including extra projection layers).
    int logical_num_layers; // The number of layers as specified by the configuration (i.e. neurons_per_layer count).
    // (Optional) Additional metadata can be added here.
} Network;

// Function prototypes.
Network* create_network(const Config *cfg);
Matrix* network_forward(Network *net, const Matrix *input, int stochastic);
double network_total_kl(Network *net);
void free_network(Network *net);
Matrix* network_backward(Network *net, const Matrix *grad_output, const Config *cfg);


#endif // NETWORK_H

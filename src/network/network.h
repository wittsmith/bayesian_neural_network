#ifndef NETWORK_H
#define NETWORK_H

#include "../config/config.h"
#include "../utils/math_utils.h"

// Abstract layer interface.
typedef struct Layer {
    void *layer;  // Pointer to the concrete layer (e.g., BayesianLinear*, BayesianConv*, DropoutLayer*, StochasticActivation*, etc.)
    // Forward pass function: given a layer pointer, input Matrix, and stochastic flag, returns an output Matrix.
    Matrix* (*forward)(void *layer, const Matrix *input, int stochastic);
    // KL divergence function: returns the KL divergence for the layer.
    double (*kl)(void *layer);
    // Cleanup function: frees the concrete layer.
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

#endif // NETWORK_H

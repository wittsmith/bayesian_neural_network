#include "optimizer.h"
#include "../network/layers/bayesian_linear.h"
#include <stdio.h>

// Update function for BayesianLinear layers using SGD.
void update_bayesian_linear(BayesianLinear *layer, double lr) {
    int total_weights = layer->output_dim * layer->input_dim;
    
    // Debug: print some parameter and gradient values before update.
    // printf("Before update: W_mean[0] = %f, dW_mean[0] = %f\n", 
    //        layer->W_mean->data[0], layer->dW_mean->data[0]);
    // printf("Before update: b_mean[0] = %f, db_mean[0] = %f\n", 
    //        layer->b_mean[0], layer->db_mean[0]);
    
    // Update weight parameters.
    for (int i = 0; i < total_weights; i++) {
        layer->W_mean->data[i] -= lr * layer->dW_mean->data[i];
    }
    // Update bias parameters.
    for (int i = 0; i < layer->output_dim; i++) {
        layer->b_mean[i] -= lr * layer->db_mean[i];
    }
    
    // Debug: print parameters after update.
    // printf("After update: W_mean[0] = %f\n", layer->W_mean->data[0]);
    // printf("After update: b_mean[0] = %f\n", layer->b_mean[0]);
    // fflush(stdout);
}


// Iterate over each layer in the network and update its parameters.
void network_update_params(Network *net, const Config *cfg) {
    for (int i = 0; i < net->num_layers; i++) {
        // Here we assume the layer is a BayesianLinear layer.
        // In a more complete implementation, you would use some type tagging
        // or function pointers to update different layer types appropriately.
        BayesianLinear *bl = (BayesianLinear*) net->layers[i]->layer;
        if (bl) {
            update_bayesian_linear(bl, cfg->learning_rate);
        } else {
            // If other types exist, call their update function here.
            printf("Warning: Unknown layer type encountered during update.\n");
        }
    }
}

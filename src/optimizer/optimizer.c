#include "optimizer.h"
#include "../network/layers/bayesian_linear.h"
#include "../network/layers/stochastic_activation.h"
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

// Update function for StochasticActivation layers using SGD.
void update_stochastic_activation(StochasticActivation *layer, double lr) {
    if (!layer) {
        printf("Warning: Invalid StochasticActivation layer in update.\n");
        return;
    }
    
    // Update the alpha_mean parameter using the stored gradient
    layer->alpha_mean -= lr * layer->d_alpha_mean;
    
    // Reset the gradient after update
    layer->d_alpha_mean = 0.0;
}

// Calculate the decayed learning rate based on the current epoch
double calculate_decayed_lr(const Config *cfg, int current_epoch) {
    if (cfg->lr_decay <= 0.0) {
        return cfg->learning_rate;  // No decay
    }
    
    // Exponential decay: lr = initial_lr * (1 / (1 + decay * epoch))
    return cfg->learning_rate / (1.0 + cfg->lr_decay * current_epoch);
}

// Iterate over each layer in the network and update its parameters.
void network_update_params(Network *net, const Config *cfg, int current_epoch) {
    // Calculate the decayed learning rate
    double decayed_lr = calculate_decayed_lr(cfg, current_epoch);
    
    for (int i = 0; i < net->num_layers; i++) {
        printf("Updating layer %d with decayed lr: %f\n", i, decayed_lr);
        
        // Handle different layer types
        switch (net->layers[i]->type) {
            case LAYER_BAYESIAN_LINEAR:
                update_bayesian_linear((BayesianLinear*)net->layers[i]->layer, decayed_lr);
                break;
                
            case LAYER_STOCHASTIC_ACTIVATION:
                update_stochastic_activation((StochasticActivation*)net->layers[i]->layer, decayed_lr);
                break;
                
            case LAYER_DROPOUT:
                // Dropout layers don't have learnable parameters
                break;
                
            case LAYER_BAYESIAN_CONV:
                // TODO: Implement update_bayesian_conv
                printf("Warning: BayesianConv layer updates not yet implemented.\n");
                break;
                
            case LAYER_PROJECTION:
                // Projection layers don't have learnable parameters
                break;
                
            default:
                printf("Warning: Unknown layer type %d encountered during update.\n", 
                       net->layers[i]->type);
                break;
        }
    }
}

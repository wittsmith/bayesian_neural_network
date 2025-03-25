#include "bayesian_conv.h"
#include "../utils/utils.h"
#include "../utils/random_utils.h"
#include "../bnn_util.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Create a new Tensor with given dimensions.
Tensor* create_tensor(int channels, int height, int width) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        handle_error("Failed to allocate Tensor structure.");
    }
    t->channels = channels;
    t->height = height;
    t->width = width;
    t->data = (double*)calloc(channels * height * width, sizeof(double));
    if (!t->data) {
        free(t);
        handle_error("Failed to allocate Tensor data.");
    }
    return t;
}

// Free a Tensor.
void free_tensor(Tensor *t) {
    if (t) {
        free(t->data);
        free(t);
    }
}

// Create a new Bayesian Convolutional layer.
BayesianConv* create_bayesian_conv(int input_channels, int output_channels, int kernel_height, int kernel_width) {
    BayesianConv *layer = (BayesianConv*)malloc(sizeof(BayesianConv));
    if (!layer) {
        handle_error("Failed to allocate BayesianConv layer.");
    }
    layer->input_channels = input_channels;
    layer->output_channels = output_channels;
    layer->kernel_height = kernel_height;
    layer->kernel_width = kernel_width;
    
    int weight_size = output_channels * input_channels * kernel_height * kernel_width;
    layer->W_mean = (double*)malloc(sizeof(double) * weight_size);
    layer->W_logvar = (double*)malloc(sizeof(double) * weight_size);
    if (!layer->W_mean || !layer->W_logvar) {
        handle_error("Failed to allocate convolutional weight arrays.");
    }
    
    layer->b_mean = (double*)malloc(sizeof(double) * output_channels);
    layer->b_logvar = (double*)malloc(sizeof(double) * output_channels);
    if (!layer->b_mean || !layer->b_logvar) {
        handle_error("Failed to allocate convolutional bias arrays.");
    }
    
    // Initialize weights and biases.
    for (int oc = 0; oc < output_channels; oc++) {
        for (int ic = 0; ic < input_channels; ic++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int idx = oc * (input_channels * kernel_height * kernel_width)
                              + ic * (kernel_height * kernel_width)
                              + kh * kernel_width + kw;
                    layer->W_mean[idx] = random_gaussian(0.0, 0.1);
                    layer->W_logvar[idx] = -5.0;
                }
            }
        }
        layer->b_mean[oc] = random_gaussian(0.0, 0.1);
        layer->b_logvar[oc] = -5.0;
    }
    
    // Initialize the Prior and Posterior pointers to NULL.
    layer->prior = NULL;
    layer->posterior = NULL;
    
    return layer;
}

// Free a Bayesian Convolutional layer.
void free_bayesian_conv(BayesianConv *layer) {
    if (layer) {
        free(layer->W_mean);
        free(layer->W_logvar);
        free(layer->b_mean);
        free(layer->b_logvar);
        free(layer);
    }
}

// Forward pass for the Bayesian convolutional layer (stride 1, no padding).
Tensor* bayesian_conv_forward(BayesianConv *layer, const Tensor *input, int stochastic) {
    if (input->channels != layer->input_channels) {
        handle_error("Input channel mismatch in bayesian_conv_forward.");
    }
    
    int out_height = input->height - layer->kernel_height + 1;
    int out_width = input->width - layer->kernel_width + 1;
    if (out_height <= 0 || out_width <= 0) {
        handle_error("Invalid output dimensions in bayesian_conv_forward.");
    }
    
    // Create output tensor: shape (output_channels, out_height, out_width)
    Tensor *output = create_tensor(layer->output_channels, out_height, out_width);
    
    // For each output channel, compute the convolution.
    for (int oc = 0; oc < layer->output_channels; oc++) {
        // Compute effective bias for this output channel.
        double b_effective;
        if (stochastic) {
            if (layer->posterior != NULL) {
                b_effective = layer->posterior->sample(layer->posterior, layer->b_mean[oc], layer->b_logvar[oc]);
            } else {
                b_effective = sample_gaussian(layer->b_mean[oc], layer->b_logvar[oc]);
            }
        } else {
            b_effective = layer->b_mean[oc];
        }
        
        // Loop over each output pixel.
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                double sum = 0.0;
                // Sum over input channels and kernel window.
                for (int ic = 0; ic < layer->input_channels; ic++) {
                    for (int kh = 0; kh < layer->kernel_height; kh++) {
                        for (int kw = 0; kw < layer->kernel_width; kw++) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            int input_idx = ic * (input->height * input->width) + ih * input->width + iw;
                            
                            int weight_idx = oc * (layer->input_channels * layer->kernel_height * layer->kernel_width)
                                             + ic * (layer->kernel_height * layer->kernel_width)
                                             + kh * layer->kernel_width + kw;
                            
                            double weight_effective;
                            if (stochastic) {
                                if (layer->posterior != NULL) {
                                    weight_effective = layer->posterior->sample(layer->posterior,
                                                                                 layer->W_mean[weight_idx],
                                                                                 layer->W_logvar[weight_idx]);
                                } else {
                                    weight_effective = sample_gaussian(layer->W_mean[weight_idx],
                                                                       layer->W_logvar[weight_idx]);
                                }
                            } else {
                                weight_effective = layer->W_mean[weight_idx];
                            }
                            
                            sum += input->data[input_idx] * weight_effective;
                        }
                    }
                }
                sum += b_effective;
                int output_idx = oc * (out_height * out_width) + oh * out_width + ow;
                output->data[output_idx] = sum;
            }
        }
    }
    
    return output;
}

// Compute the total KL divergence for the layer's weights and biases using the Prior interface.
// For each weight and bias, if a Prior is assigned, call its compute_kl() function;
// otherwise, use a default Gaussian KL divergence with variance 1.0.
double bayesian_conv_kl(BayesianConv *layer) {
    double kl_total = 0.0;
    int total_weights = layer->output_channels * layer->input_channels * layer->kernel_height * layer->kernel_width;
    
    if (layer->prior == NULL) {
        double default_variance = 1.0;
        for (int i = 0; i < total_weights; i++) {
            kl_total += kl_divergence_single(layer->W_mean[i], layer->W_logvar[i], default_variance);
        }
        for (int i = 0; i < layer->output_channels; i++) {
            kl_total += kl_divergence_single(layer->b_mean[i], layer->b_logvar[i], default_variance);
        }
    } else {
        for (int i = 0; i < total_weights; i++) {
            kl_total += layer->prior->compute_kl(layer->prior, layer->W_mean[i], layer->W_logvar[i]);
        }
        for (int i = 0; i < layer->output_channels; i++) {
            kl_total += layer->prior->compute_kl(layer->prior, layer->b_mean[i], layer->b_logvar[i]);
        }
    }
    return kl_total;
}

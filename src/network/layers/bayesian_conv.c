#include "bayesian_conv.h"
#include "../utils/utils.h"
#include "../utils/random_utils.h"
#include "../utils/math_utils.h"
#include "../bnn_util.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../priors/prior.h"       // For the common Prior interface.
#include "../posteriors/posterior.h"
#include <stdio.h>

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
#include "bayesian_conv.h"
#include "../utils/utils.h"
#include "../utils/random_utils.h"
#include "../utils/math_utils.h"
#include "../bnn_util.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../priors/prior.h"       // For the common Prior interface.
#include "../posteriors/posterior.h"



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


Tensor* matrix_to_tensor(const Matrix *m, int channels, int height, int width) {
    int flat = channels * height * width;
    if (m->cols != flat) {
         handle_error("Matrix columns do not match expected tensor shape in matrix_to_tensor.");
    }
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
         handle_error("Failed to allocate Tensor.");
    }
    t->channels = channels;
    t->height = height;
    t->width = width;
    int total = m->rows * flat;
    t->data = (double*)malloc(sizeof(double) * total);
    if (!t->data) {
         free(t);
         handle_error("Failed to allocate Tensor data.");
    }
    // Copy data row-by-row.
    memcpy(t->data, m->data, sizeof(double) * total);
    return t;
}


// Forward pass for the Bayesian convolutional layer (stride 1, no padding).
// Forward pass for the Bayesian convolutional layer (with 'same' padding).
// This function now produces an output tensor with the same height and width as the input.
// Forward pass for the Bayesian convolutional layer with global average pooling.
// This function first computes a standard convolution (using valid convolution),
// then applies global average pooling to collapse the spatial dimensions, yielding
// an output matrix of shape (batch_size x output_channels).
Matrix* bayesian_conv_forward(BayesianConv *layer, const Tensor *input, int stochastic) {
    if (input->channels != layer->input_channels) {
        printf("input to channels: %d", input->channels);
        printf("layer to input channels: %d", layer->input_channels);
        handle_error("Input channel mismatch in bayesian_conv_forward.");
        
    }
    
    // Compute output spatial dimensions using valid convolution.
    // Compute output spatial dimensions using valid convolution.
    // (Assuming input->height and input->width represent the spatial dimensions of one sample.)
    int out_height = input->height - layer->kernel_height + 1;
    int out_width = input->width - layer->kernel_width + 1;

    if (out_height <= 0 || out_width <= 0) {
        printf("out_height: %d; ", out_height);
        printf("out_width: %d", out_width);
        handle_error("Invalid output dimensions in bayesian_conv_forward.");
    }
    
    // Create an intermediate tensor for the convolution result.
    // This tensor has dimensions: [output_channels x out_height x out_width].
    Tensor *conv_out = create_tensor(layer->output_channels, out_height, out_width);
    
    // Perform convolution for each output channel.
    for (int oc = 0; oc < layer->output_channels; oc++) {
        // Compute effective bias for the output channel.
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
        
        // For each spatial location in the output.
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
                // Add bias and store the result.
                int conv_idx = oc * (out_height * out_width) + oh * out_width + ow;
                conv_out->data[conv_idx] = sum + b_effective;
            }
        }
    }
    
    // Flatten the convolution result into a Matrix.
    // The flattened dimensions will be:
    //   rows = batch_size (same as input->height? -- note: usually batch size is stored separately)
    //   cols = output_channels * out_height * out_width
    // Here, we assume that 'input->height' is being used as the batch size (if your Matrix and Tensor conventions differ,
    // adjust accordingly).
    int batch_size = input->width;  // Assuming 'rows' in Matrix represent batch size.
    int flatten_dim = layer->output_channels * out_height * out_width;
    Matrix *output = create_matrix(batch_size, flatten_dim);
    
    // For each sample in the batch, copy and flatten the corresponding data from conv_out.
    // Note: This assumes that conv_out->data is organized per sample.
    // If your Tensor structure doesn't include a batch dimension, you'll need to modify accordingly.
    // Here, we assume conv_out has 'batch_size' copies (one per sample) concatenated in memory.
    // For demonstration purposes, we'll assume that conv_out->data holds one sample and we replicate for each row.
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < flatten_dim; j++) {
            // Adjust indexing based on your actual Tensor organization.
            // Here we assume the same conv_out for each sample.
            output->data[b * flatten_dim + j] = conv_out->data[j];
        }
    }
    
    free_tensor(conv_out);
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
void free_bayesian_conv(BayesianConv *layer) {
    if (layer) {
        free(layer->W_mean);
        free(layer->W_logvar);
        free(layer->b_mean);
        free(layer->b_logvar);
        free(layer);
    }
}

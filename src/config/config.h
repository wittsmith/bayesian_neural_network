#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>

//====================//
// Default Constants  //
//====================//

// Optimization & Training
#define DEFAULT_LEARNING_RATE         0.001
#define DEFAULT_MINI_BATCH_SIZE       32
#define DEFAULT_NUM_EPOCHS            100
#define DEFAULT_LR_DECAY              0.0         // No decay by default
#define DEFAULT_OPTIMIZER             0           // 0: SGD, 1: Adam, etc.
#define DEFAULT_GRAD_CLIP             5.0
#define DEFAULT_NOISE_INJECTION       0.0

// Adam Optimizer Specific
#define DEFAULT_ADAM_BETA1           0.9
#define DEFAULT_ADAM_BETA2           0.999
#define DEFAULT_ADAM_EPSILON         1e-8

// Inference Method
#define DEFAULT_INFERENCE_METHOD      0           // 0: Bayes-by-backprop, 1: MCMC, 2: SGLD, 3: MC-Dropout, 4: EP

// Architecture (High-Level)
#define DEFAULT_NUM_LAYERS            3
#define DEFAULT_NEURONS_PER_LAYER     "128,128,10" // Comma-separated list
#define DEFAULT_LAYER_TYPES           "linear,linear,linear"  // Comma-separated list of layer types (e.g., "linear,conv,dropout")
#define DEFAULT_WEIGHT_INIT_METHOD    0           // 0: Xavier, 1: He, etc.
#define DEFAULT_INPUT_DIM            100          // Default input dimension

// Prior Distribution
#define DEFAULT_PRIOR_TYPE            0           // 0: Gaussian, 1: Laplace, 2: Mixture
#define DEFAULT_PRIOR_VARIANCE        1.0
#define DEFAULT_COVARIANCE_STRUCTURE  0           // 0: Mean-field, 1: Full covariance

// Posterior Approximation Method
#define DEFAULT_POSTERIOR_METHOD      0           // 0: Mean-field, 1: Structured, 2: Flipout

// Variational Inference (BBB)
#define DEFAULT_MC_SAMPLES_TRAIN      1           // MC samples per gradient update
#define DEFAULT_KL_WEIGHT             .001        // Scaling factor for KL divergence
#define DEFAULT_LOCAL_REPARAM         1           // Flag: 1 to use local reparameterization

// MC-Dropout
#define DEFAULT_DROPOUT_PROB          0.5
#define DEFAULT_MC_SAMPLES_INFERENCE  10          // Number of test-time forward passes

// MCMC Methods
#define DEFAULT_MCMC_STEP_SIZE        0.001
#define DEFAULT_MCMC_BURN_IN          1000
#define DEFAULT_MCMC_NOISE            0.01

// Expectation Propagation (EP)
#define DEFAULT_EP_DAMPING            0.5
#define DEFAULT_EP_ITERATIONS         100
#define DEFAULT_EP_TOLERANCE          1e-4

// Sampling / Predictive Inference
#define DEFAULT_SAMPLING_TEMPERATURE  1.0

// Regularization
#define DEFAULT_REGULARIZATION_WEIGHT 0.0001
#define DEFAULT_KL_ANNEALING          0           // 0: disabled, 1: enabled

// BBB-specific extras
#define DEFAULT_BBB_LEARN_VARIANCE    1           // 1: learn variance, 0: fixed
#define DEFAULT_BBB_NOISE_SCALING     1.0

// Ensemble-specific (if applicable)
#define DEFAULT_ENSEMBLE_SIZE         1

// Data
#define DEFAULT_DATA_PATH             "data/train.csv"

//=====================//
// Config Structure    //
//=====================//

typedef struct {
    // Optimization & Training
    double learning_rate;
    int mini_batch_size;
    int num_epochs;
    double lr_decay;
    int optimizer;
    double grad_clip;
    double noise_injection;
    
    // Adam Optimizer Specific
    double adam_beta1;
    double adam_beta2;
    double adam_epsilon;
    
    // Inference Method
    int inference_method;
    
    // Architecture (High-Level)
    int num_layers;
    char neurons_per_layer[256]; // Comma-separated list of neuron counts
    char layer_types[256];       // Comma-separated list of layer types
    int weight_init_method;
    int input_dim;              // Input dimension for the network
    
    // Prior Distribution
    int prior_type;
    double prior_variance;
    int covariance_structure;
    
    // Posterior Approximation Method
    int posterior_method;  // 0: Mean-field, 1: Structured, 2: Flipout, etc.
    
    // Variational Inference (BBB)
    int mc_samples_train;
    double kl_weight;
    int local_reparam;
    
    // MC-Dropout
    double dropout_prob;
    int mc_samples_inference;
    
    // MCMC Methods
    double mcmc_step_size;
    int mcmc_burn_in;
    double mcmc_noise;
    
    // EP Methods
    double ep_damping;
    int ep_iterations;
    double ep_tolerance;
    
    // Sampling / Predictive Inference
    double sampling_temperature;
    
    // Regularization
    double regularization_weight;
    int kl_annealing;
    
    // BBB-specific extras
    int bbb_learn_variance;
    double bbb_noise_scaling;
    
    // Ensemble-specific
    int ensemble_size;
    
    // Data
    char data_path[256];
} Config;

//=====================//
// Function Prototypes //
//=====================//

// Initialization and parsing
void init_config(Config *cfg);
void parse_args(Config *cfg, int argc, char *argv[]);
int load_config_file(Config *cfg, const char *filename);

// Getters (for encapsulation if needed)
double get_learning_rate(const Config *cfg);
int get_mini_batch_size(const Config *cfg);
int get_num_epochs(const Config *cfg);
double get_dropout_prob(const Config *cfg);
double get_regularization_weight(const Config *cfg);
double get_prior_variance(const Config *cfg);
int get_inference_method(const Config *cfg);
const char* get_data_path(const Config *cfg);
double get_kl_weight(const Config *cfg);

#endif // CONFIG_H

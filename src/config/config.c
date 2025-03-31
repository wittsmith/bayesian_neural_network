#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Initialize configuration with default values.
void init_config(Config *cfg) {
    if (!cfg) return;
    
    // Optimization & Training
    cfg->learning_rate     = DEFAULT_LEARNING_RATE;
    cfg->mini_batch_size   = DEFAULT_MINI_BATCH_SIZE;
    cfg->num_epochs        = DEFAULT_NUM_EPOCHS;
    cfg->lr_decay          = DEFAULT_LR_DECAY;
    cfg->optimizer         = DEFAULT_OPTIMIZER;
    cfg->grad_clip         = DEFAULT_GRAD_CLIP;
    cfg->noise_injection   = DEFAULT_NOISE_INJECTION;
    
    // Inference Method
    cfg->inference_method  = DEFAULT_INFERENCE_METHOD;
    
    // Architecture
    cfg->num_layers        = DEFAULT_NUM_LAYERS;
    strncpy(cfg->neurons_per_layer, DEFAULT_NEURONS_PER_LAYER, sizeof(cfg->neurons_per_layer)-1);
    cfg->neurons_per_layer[sizeof(cfg->neurons_per_layer)-1] = '\0';
    strncpy(cfg->layer_types, DEFAULT_LAYER_TYPES, sizeof(cfg->layer_types)-1);
    cfg->layer_types[sizeof(cfg->layer_types)-1] = '\0';
    cfg->weight_init_method = DEFAULT_WEIGHT_INIT_METHOD;
    cfg->input_dim         = DEFAULT_INPUT_DIM;
    
    // Prior Distribution
    cfg->prior_type        = DEFAULT_PRIOR_TYPE;
    cfg->prior_variance    = DEFAULT_PRIOR_VARIANCE;
    cfg->covariance_structure = DEFAULT_COVARIANCE_STRUCTURE;
    
    // Posterior Approximation Method
    cfg->posterior_method  = DEFAULT_POSTERIOR_METHOD;
    
    // Variational Inference (BBB)
    cfg->mc_samples_train  = DEFAULT_MC_SAMPLES_TRAIN;
    cfg->kl_weight         = DEFAULT_KL_WEIGHT;
    cfg->local_reparam     = DEFAULT_LOCAL_REPARAM;
    
    // MC-Dropout
    cfg->dropout_prob      = DEFAULT_DROPOUT_PROB;
    cfg->mc_samples_inference = DEFAULT_MC_SAMPLES_INFERENCE;
    
    // MCMC Methods
    cfg->mcmc_step_size    = DEFAULT_MCMC_STEP_SIZE;
    cfg->mcmc_burn_in      = DEFAULT_MCMC_BURN_IN;
    cfg->mcmc_noise        = DEFAULT_MCMC_NOISE;
    
    // EP Methods
    cfg->ep_damping        = DEFAULT_EP_DAMPING;
    cfg->ep_iterations     = DEFAULT_EP_ITERATIONS;
    cfg->ep_tolerance      = DEFAULT_EP_TOLERANCE;
    
    // Sampling / Predictive Inference
    cfg->sampling_temperature = DEFAULT_SAMPLING_TEMPERATURE;
    
    // Regularization
    cfg->regularization_weight = DEFAULT_REGULARIZATION_WEIGHT;
    cfg->kl_annealing      = DEFAULT_KL_ANNEALING;
    
    // BBB-specific extras
    cfg->bbb_learn_variance = DEFAULT_BBB_LEARN_VARIANCE;
    cfg->bbb_noise_scaling  = DEFAULT_BBB_NOISE_SCALING;
    
    // Ensemble-specific
    cfg->ensemble_size      = DEFAULT_ENSEMBLE_SIZE;
    
    // Data
    strncpy(cfg->data_path, DEFAULT_DATA_PATH, sizeof(cfg->data_path)-1);
    cfg->data_path[sizeof(cfg->data_path)-1] = '\0';
}

// Parse command-line arguments to override default configuration.
// Example usage: --lr 0.005 --batch 64 --epochs 200 --dropout 0.4 --reg 0.00005 --priorvar 2.0 
// --inference 1 --data "data/new_train.csv" --layers 4 --neurons "256,128,64,10" --layer_types "linear,conv,dropout" etc.
void parse_args(Config *cfg, int argc, char *argv[]) {
    if (!cfg) return;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) {
            cfg->learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) {
            cfg->mini_batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) {
            cfg->num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr_decay") == 0 && i+1 < argc) {
            cfg->lr_decay = atof(argv[++i]);
        } else if (strcmp(argv[i], "--optimizer") == 0 && i+1 < argc) {
            cfg->optimizer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--grad_clip") == 0 && i+1 < argc) {
            cfg->grad_clip = atof(argv[++i]);
        } else if (strcmp(argv[i], "--input_dim") == 0 && i+1 < argc) {
            cfg->input_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--noise_injection") == 0 && i+1 < argc) {
            cfg->noise_injection = atof(argv[++i]);
        } else if (strcmp(argv[i], "--inference") == 0 && i+1 < argc) {
            cfg->inference_method = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i+1 < argc) {
            cfg->num_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--neurons") == 0 && i+1 < argc) {
            strncpy(cfg->neurons_per_layer, argv[++i], sizeof(cfg->neurons_per_layer)-1);
            cfg->neurons_per_layer[sizeof(cfg->neurons_per_layer)-1] = '\0';
        } else if (strcmp(argv[i], "--layer_types") == 0 && i+1 < argc) {
            strncpy(cfg->layer_types, argv[++i], sizeof(cfg->layer_types)-1);
            cfg->layer_types[sizeof(cfg->layer_types)-1] = '\0';
        } else if (strcmp(argv[i], "--weight_init") == 0 && i+1 < argc) {
            cfg->weight_init_method = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prior_type") == 0 && i+1 < argc) {
            cfg->prior_type = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--priorvar") == 0 && i+1 < argc) {
            cfg->prior_variance = atof(argv[++i]);
        } else if (strcmp(argv[i], "--cov_structure") == 0 && i+1 < argc) {
            cfg->covariance_structure = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--posterior_method") == 0 && i+1 < argc) {
            cfg->posterior_method = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mc_samples_train") == 0 && i+1 < argc) {
            cfg->mc_samples_train = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--kl_weight") == 0 && i+1 < argc) {
            cfg->kl_weight = atof(argv[++i]);
        } else if (strcmp(argv[i], "--local_reparam") == 0 && i+1 < argc) {
            cfg->local_reparam = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dropout") == 0 && i+1 < argc) {
            cfg->dropout_prob = atof(argv[++i]);
        } else if (strcmp(argv[i], "--mc_samples_inference") == 0 && i+1 < argc) {
            cfg->mc_samples_inference = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mcmc_step_size") == 0 && i+1 < argc) {
            cfg->mcmc_step_size = atof(argv[++i]);
        } else if (strcmp(argv[i], "--mcmc_burn_in") == 0 && i+1 < argc) {
            cfg->mcmc_burn_in = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mcmc_noise") == 0 && i+1 < argc) {
            cfg->mcmc_noise = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ep_damping") == 0 && i+1 < argc) {
            cfg->ep_damping = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ep_iter") == 0 && i+1 < argc) {
            cfg->ep_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ep_tol") == 0 && i+1 < argc) {
            cfg->ep_tolerance = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sampling_temp") == 0 && i+1 < argc) {
            cfg->sampling_temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--reg") == 0 && i+1 < argc) {
            cfg->regularization_weight = atof(argv[++i]);
        } else if (strcmp(argv[i], "--kl_annealing") == 0 && i+1 < argc) {
            cfg->kl_annealing = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bbb_learnvar") == 0 && i+1 < argc) {
            cfg->bbb_learn_variance = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bbb_noise") == 0 && i+1 < argc) {
            cfg->bbb_noise_scaling = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ensemble_size") == 0 && i+1 < argc) {
            cfg->ensemble_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) {
            strncpy(cfg->data_path, argv[++i], sizeof(cfg->data_path)-1);
            cfg->data_path[sizeof(cfg->data_path)-1] = '\0';
        } else {
            printf("Unknown argument: %s\n", argv[i]);
        }
    }
}

// Load configuration from a file (with key=value pairs).
// Returns 0 on success, -1 on failure.
int load_config_file(Config *cfg, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Could not open config file: %s\n", filename);
        return -1;
    }
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        char key[64], value[64];
        if (sscanf(line, "%63[^=]=%63s", key, value) == 2) {
            if (strcmp(key, "learning_rate") == 0) {
                cfg->learning_rate = atof(value);
            } else if (strcmp(key, "mini_batch_size") == 0) {
                cfg->mini_batch_size = atoi(value);
            } else if (strcmp(key, "num_epochs") == 0) {
                cfg->num_epochs = atoi(value);
            } else if (strcmp(key, "lr_decay") == 0) {
                cfg->lr_decay = atof(value);
            } else if (strcmp(key, "optimizer") == 0) {
                cfg->optimizer = atoi(value);
            } else if (strcmp(key, "grad_clip") == 0) {
                cfg->grad_clip = atof(value);
            } else if (strcmp(key, "noise_injection") == 0) {
                cfg->noise_injection = atof(value);
            } else if (strcmp(key, "inference_method") == 0) {
                cfg->inference_method = atoi(value);
            } else if (strcmp(key, "num_layers") == 0) {
                cfg->num_layers = atoi(value);
            } else if (strcmp(key, "neurons_per_layer") == 0) {
                strncpy(cfg->neurons_per_layer, value, sizeof(cfg->neurons_per_layer)-1);
                cfg->neurons_per_layer[sizeof(cfg->neurons_per_layer)-1] = '\0';
            } else if (strcmp(key, "layer_types") == 0) {
                strncpy(cfg->layer_types, value, sizeof(cfg->layer_types)-1);
                cfg->layer_types[sizeof(cfg->layer_types)-1] = '\0';
            } else if (strcmp(key, "weight_init_method") == 0) {
                cfg->weight_init_method = atoi(value);
            } else if (strcmp(key, "prior_type") == 0) {
                cfg->prior_type = atoi(value);
            } else if (strcmp(key, "prior_variance") == 0) {
                cfg->prior_variance = atof(value);
            } else if (strcmp(key, "covariance_structure") == 0) {
                cfg->covariance_structure = atoi(value);
            } else if (strcmp(key, "posterior_method") == 0) {
                cfg->posterior_method = atoi(value);
            } else if (strcmp(key, "mc_samples_train") == 0) {
                cfg->mc_samples_train = atoi(value);
            } else if (strcmp(key, "kl_weight") == 0) {
                cfg->kl_weight = atof(value);
            } else if (strcmp(key, "local_reparam") == 0) {
                cfg->local_reparam = atoi(value);
            } else if (strcmp(key, "dropout_prob") == 0) {
                cfg->dropout_prob = atof(value);
            } else if (strcmp(key, "mc_samples_inference") == 0) {
                cfg->mc_samples_inference = atoi(value);
            } else if (strcmp(key, "mcmc_step_size") == 0) {
                cfg->mcmc_step_size = atof(value);
            } else if (strcmp(key, "mcmc_burn_in") == 0) {
                cfg->mcmc_burn_in = atoi(value);
            } else if (strcmp(key, "mcmc_noise") == 0) {
                cfg->mcmc_noise = atof(value);
            } else if (strcmp(key, "ep_damping") == 0) {
                cfg->ep_damping = atof(value);
            } else if (strcmp(key, "ep_iterations") == 0) {
                cfg->ep_iterations = atoi(value);
            } else if (strcmp(key, "ep_tolerance") == 0) {
                cfg->ep_tolerance = atof(value);
            } else if (strcmp(key, "sampling_temperature") == 0) {
                cfg->sampling_temperature = atof(value);
            } else if (strcmp(key, "regularization_weight") == 0) {
                cfg->regularization_weight = atof(value);
            } else if (strcmp(key, "kl_annealing") == 0) {
                cfg->kl_annealing = atoi(value);
            } else if (strcmp(key, "bbb_learn_variance") == 0) {
                cfg->bbb_learn_variance = atoi(value);
            } else if (strcmp(key, "bbb_noise_scaling") == 0) {
                cfg->bbb_noise_scaling = atof(value);
            } else if (strcmp(key, "ensemble_size") == 0) {
                cfg->ensemble_size = atoi(value);
            } else if (strcmp(key, "data_path") == 0) {
                strncpy(cfg->data_path, value, sizeof(cfg->data_path)-1);
                cfg->data_path[sizeof(cfg->data_path)-1] = '\0';
            }
        }
    }
    fclose(fp);
    return 0;
}

//=====================//
// Getters             //
//=====================//

double get_learning_rate(const Config *cfg) {
    return cfg->learning_rate;
}

int get_mini_batch_size(const Config *cfg) {
    return cfg->mini_batch_size;
}

int get_num_epochs(const Config *cfg) {
    return cfg->num_epochs;
}

double get_dropout_prob(const Config *cfg) {
    return cfg->dropout_prob;
}

double get_regularization_weight(const Config *cfg) {
    return cfg->regularization_weight;
}

double get_prior_variance(const Config *cfg) {
    return cfg->prior_variance;
}

int get_inference_method(const Config *cfg) {
    return cfg->inference_method;
}

const char* get_data_path(const Config *cfg) {
    return cfg->data_path;
}

double get_kl_weight(const Config *cfg) {
    return cfg->kl_weight;
}
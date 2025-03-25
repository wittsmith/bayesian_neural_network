CC = gcc
CFLAGS = -I./config -I./layers -I./priors -I./posteriors -I./utils -I./network -Wall -O2
COMMON_SOURCES = config/config.c utils/utils.c utils/math_utils.c utils/random_utils.c network/bnn_util.c
LAYER_SOURCES = network/layers/bayesian_linear.c network/layers/bayesian_conv.c network/layers/dropout_layer.c network/layers/stochastic_activation.c
PRIOR_SOURCES = network/priors/prior_laplace.c network/priors/prior_mixture.c
POSTERIOR_SOURCES = network/posteriors/posterior_flipout.c network/posteriors/posterior_structured.c
NETWORK_SOURCES = network/network.c

# Test targets:
TEST_NETWORK = tests/test_network.c
TEST_LAYERS = tests/test_layers.c

all: test_network test_layers

test_network:
	$(CC) $(CFLAGS) $(COMMON_SOURCES) $(LAYER_SOURCES) $(PRIOR_SOURCES) $(POSTERIOR_SOURCES) $(NETWORK_SOURCES) $(TEST_NETWORK) -lm -o test_network

test_layers:
	$(CC) $(CFLAGS) $(COMMON_SOURCES) $(LAYER_SOURCES) $(PRIOR_SOURCES) $(POSTERIOR_SOURCES) $(NETWORK_SOURCES) $(TEST_LAYERS) -lm -o test_layers

clean:
	rm -f test_network test_layers

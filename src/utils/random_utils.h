#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

// Initialize the random number generator with a given seed.
void init_random(unsigned int seed);

// Return a random double in the range [0, 1)
double random_uniform();

// Generate a normally distributed random number using the Box-Muller transform.
double random_gaussian(double mean, double stddev);

// Return 1 with probability p, 0 otherwise.
int random_bernoulli(double p);

#endif // RANDOM_UTILS_H

#ifndef BNN_UTIL_H
#define BNN_UTIL_H

// Reparameterization and KL divergence helper functions for BNN layers.

// sample_gaussian:
//   Returns a sample from a Gaussian distribution using the reparameterization trick.
//   Given a mean and log-variance, it computes: sample = mean + exp(0.5 * logvar) * epsilon,
//   where epsilon ~ N(0,1).
double sample_gaussian(double mean, double logvar);

// kl_divergence_single:
//   Computes the KL divergence between the approximate posterior N(mu, sigma^2) and the prior
//   N(0, prior_variance). Here sigma^2 is computed as exp(logvar).
//   The formula used is:
//      KL = 0.5 * ( (exp(logvar) + mu^2) / prior_variance - 1 + log(prior_variance) - logvar )
double kl_divergence_single(double mu, double logvar, double prior_variance);

// compute_total_kl_divergence:
//   Given arrays of means and log-variances (of length 'length'), computes the total KL divergence
//   by summing kl_divergence_single over all elements.
double compute_total_kl_divergence(const double *mu, const double *logvar, int length, double prior_variance);

#endif // BNN_UTIL_H

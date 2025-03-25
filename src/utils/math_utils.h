#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// A simple matrix structure for use in BNN math operations.
typedef struct {
    int rows;
    int cols;
    double *data;  // Stored in row-major order.
} Matrix;

// Matrix management
Matrix* create_matrix(int rows, int cols);
// Allocates and initializes a matrix with the given dimensions
void free_matrix(Matrix *m);

// Basic matrix operations
Matrix* matrix_multiply(const Matrix *A, const Matrix *B);
Matrix* matrix_add(const Matrix *A, const Matrix *B);
Matrix* matrix_transpose(const Matrix *A);

// Vector operations
double vector_dot(const double *a, const double *b, int length);
// Dot product of the vector
double vector_norm(const double *a, int length);
// L2 norm of a vector

// Example: Compute the KL divergence between two univariate Gaussians.
// D_KL( N(mu1, var1) || N(mu2, var2) )
double kl_divergence_gaussian(double mu1, double var1, double mu2, double var2);
// Computes KL divergence between two univariate Gaussian distributions using means and variances

#endif // MATH_UTILS_H

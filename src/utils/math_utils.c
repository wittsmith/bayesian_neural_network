#include "math_utils.h"
#include "utils.h"  // for error handling
#include <stdlib.h>
#include <math.h>
#include <string.h>

Matrix* create_matrix(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) {
        handle_error("Failed to allocate memory for matrix structure.");
    }
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));
    if (!m->data) {
        free(m);
        handle_error("Failed to allocate memory for matrix data.");
    }
    return m;
}

void free_matrix(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

Matrix* matrix_multiply(const Matrix *A, const Matrix *B) {
    if (A->cols != B->rows) {
        handle_error("Matrix multiplication dimension mismatch.");
    }
    Matrix *result = create_matrix(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
    return result;
}

Matrix* matrix_add(const Matrix *A, const Matrix *B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        handle_error("Matrix addition dimension mismatch.");
    }
    Matrix *result = create_matrix(A->rows, A->cols);
    int total = A->rows * A->cols;
    for (int i = 0; i < total; i++) {
        result->data[i] = A->data[i] + B->data[i];
    }
    return result;
}

Matrix* matrix_transpose(const Matrix *A) {
    Matrix *result = create_matrix(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            result->data[j * result->cols + i] = A->data[i * A->cols + j];
        }
    }
    return result;
}

double vector_dot(const double *a, const double *b, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

double vector_norm(const double *a, int length) {
    return sqrt(vector_dot(a, a, length));
}

double kl_divergence_gaussian(double mu1, double var1, double mu2, double var2) {
    if (var1 <= 0 || var2 <= 0) {
        handle_error("Variance must be positive for KL divergence calculation.");
    }
    // Formula for KL divergence
    return 0.5 * ((var1 / var2) + (pow(mu2 - mu1, 2) / var2) - 1 + log(var2 / var1));
}
// zero_matrix: Sets every element of the matrix to 0.0.
void zero_matrix(Matrix *m) {
    if (!m) {
        handle_error("zero_matrix: Matrix is NULL or not properly allocated.");
    }
    if (!m->data){
        handle_error("zero_matrix: matrix has no data");
    }
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i++) {
        m->data[i] = 0.0;
    }
}

// zero_array: Sets every element of the array to 0.0.
void zero_array(double *arr, int length) {
    if (!arr) {
        handle_error("zero_array: Array pointer is NULL.");
    }
    for (int i = 0; i < length; i++) {
        arr[i] = 0.0;
    }
}

// copy_matrix: Creates a deep copy of the provided matrix.
Matrix* copy_matrix(const Matrix *m) {
    if (!m) {
        handle_error("copy_matrix: Input matrix is NULL.");
    }
    Matrix *new_matrix = create_matrix(m->rows, m->cols);
    if (!new_matrix) {
        handle_error("copy_matrix: Failed to allocate memory for new matrix.");
    }
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i++) {
        new_matrix->data[i] = m->data[i];
    }
    return new_matrix;
}


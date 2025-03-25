#include "data.h"
#include "utils.h"         // For handle_error() and logging if needed.
#include "random_utils.h"  // For random_uniform()
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper: Count the number of lines (excluding header if necessary) and determine number of columns.
static void count_csv(const char *filename, int has_header, int *num_samples, int *num_columns) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        handle_error("Failed to open CSV file in count_csv().");
    }

    char line[1024];
    *num_samples = 0;
    *num_columns = 0;
    
    // Skip header if requested.
    if (has_header && fgets(line, sizeof(line), fp) != NULL) {
        // Do nothing; header is skipped.
    }
    
    // Read first data line to count columns.
    if (fgets(line, sizeof(line), fp) != NULL) {
        (*num_samples)++;
        // Count comma-separated tokens.
        int count = 0;
        char *token = strtok(line, ",");
        while (token != NULL) {
            count++;
            token = strtok(NULL, ",");
        }
        *num_columns = count;
    }
    
    // Count remaining lines.
    while (fgets(line, sizeof(line), fp) != NULL) {
        (*num_samples)++;
    }
    fclose(fp);
}

// Load CSV file into a Dataset.
// Assumes that the last column is the label.
Dataset* load_csv(const char *filename, int has_header) {
    int total_samples = 0, total_columns = 0;
    count_csv(filename, has_header, &total_samples, &total_columns);
    
    if (total_columns < 2) {
        handle_error("CSV file must contain at least two columns (features and label).");
    }
    
    int num_features = total_columns - 1;  // Last column is label.
    
    // Allocate dataset structure.
    Dataset *ds = (Dataset*)malloc(sizeof(Dataset));
    if (!ds) {
        handle_error("Failed to allocate memory for Dataset structure.");
    }
    ds->num_samples = total_samples;
    ds->num_features = num_features;
    ds->X = (double*)malloc(sizeof(double) * total_samples * num_features);
    ds->y = (double*)malloc(sizeof(double) * total_samples);
    if (!ds->X || !ds->y) {
        handle_error("Failed to allocate memory for dataset arrays.");
    }
    
    // Open file again to read data.
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        handle_error("Failed to open CSV file in load_csv().");
    }
    
    char line[1024];
    int row = 0;
    
    // Skip header if needed.
    if (has_header && fgets(line, sizeof(line), fp) == NULL) {
        handle_error("Failed to read header line from CSV file.");
    }
    
    // Read each line and parse.
    while (fgets(line, sizeof(line), fp) != NULL) {
        // Remove newline character if present.
        line[strcspn(line, "\r\n")] = 0;
        
        int col = 0;
        char *token = strtok(line, ",");
        while (token != NULL) {
            // For features: first (num_features) tokens.
            if (col < num_features) {
                ds->X[row * num_features + col] = atof(token);
            } else if (col == num_features) {
                // Last column is the label.
                ds->y[row] = atof(token);
            }
            col++;
            token = strtok(NULL, ",");
        }
        if (col != total_columns) {
            log_warn("Row %d in CSV does not have expected number of columns.", row);
        }
        row++;
    }
    fclose(fp);
    return ds;
}

// Free a dataset.
void free_dataset(Dataset *ds) {
    if (ds) {
        free(ds->X);
        free(ds->y);
        free(ds);
    }
}

// Shuffle the dataset in-place using Fisher–Yates algorithm.
void shuffle_dataset(Dataset *ds) {
    if (!ds) return;
    int n = ds->num_samples;
    int num_features = ds->num_features;
    double *temp_row = (double*)malloc(sizeof(double) * num_features);
    if (!temp_row) {
        handle_error("Memory allocation failed in shuffle_dataset().");
    }
    for (int i = n - 1; i > 0; i--) {
        // Get a random index j in [0, i]
        int j = (int)(random_uniform() * (i + 1));
        
        // Swap row i and row j in X.
        for (int k = 0; k < num_features; k++) {
            temp_row[k] = ds->X[i * num_features + k];
            ds->X[i * num_features + k] = ds->X[j * num_features + k];
            ds->X[j * num_features + k] = temp_row[k];
        }
        // Swap corresponding labels.
        double temp_label = ds->y[i];
        ds->y[i] = ds->y[j];
        ds->y[j] = temp_label;
    }
    free(temp_row);
}

// Create a mini-batch dataset from the given dataset starting at index 'start'
// and containing 'batch_size' examples. If start+batch_size exceeds the dataset size,
// it adjusts the batch size accordingly.
Dataset* get_minibatch(const Dataset *ds, int start, int batch_size) {
    if (!ds) return NULL;
    if (start < 0 || start >= ds->num_samples) {
        handle_error("Invalid start index in get_minibatch().");
    }
    
    int end = start + batch_size;
    if (end > ds->num_samples) {
        end = ds->num_samples;
        batch_size = end - start;
    }
    
    Dataset *batch = (Dataset*)malloc(sizeof(Dataset));
    if (!batch) {
        handle_error("Failed to allocate memory for mini-batch Dataset.");
    }
    batch->num_samples = batch_size;
    batch->num_features = ds->num_features;
    batch->X = (double*)malloc(sizeof(double) * batch_size * ds->num_features);
    batch->y = (double*)malloc(sizeof(double) * batch_size);
    if (!batch->X || !batch->y) {
        handle_error("Failed to allocate memory for mini-batch arrays.");
    }
    
    // Copy rows and labels from the dataset.
    memcpy(batch->X, ds->X + start * ds->num_features, sizeof(double) * batch_size * ds->num_features);
    memcpy(batch->y, ds->y + start, sizeof(double) * batch_size);
    
    return batch;
}

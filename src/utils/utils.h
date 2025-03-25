#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

// Log levels for controlling verbosity
typedef enum {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR
} LogLevel;

// Set the global log level
void set_log_level(LogLevel level);

// Logging functions
void log_debug(const char *format, ...);
void log_info(const char *format, ...);
void log_warn(const char *format, ...);
void log_error(const char *format, ...);

// Utility error handler that logs and exits
void handle_error(const char *msg);

#endif // UTILS_H

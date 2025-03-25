#include "utils.h"
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>

static LogLevel current_log_level = LOG_LEVEL_INFO;

void set_log_level(LogLevel level) {
    current_log_level = level;
}

// Internal function to log a message with a timestamp and level prefix.
static void log_message(LogLevel level, const char *prefix, const char *format, va_list args) {
    if (level < current_log_level) return;

    // Generate timestamp
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char time_buf[20];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", t);
    
    fprintf(stderr, "[%s] %s: ", time_buf, prefix);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
}

void log_debug(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_DEBUG, "DEBUG", format, args);
    va_end(args);
}

void log_info(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_INFO, "INFO", format, args);
    va_end(args);
}

void log_warn(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_WARN, "WARN", format, args);
    va_end(args);
}

void log_error(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_ERROR, "ERROR", format, args);
    va_end(args);
}

void handle_error(const char *msg) {
    log_error("Fatal error: %s", msg);
    exit(EXIT_FAILURE);
}

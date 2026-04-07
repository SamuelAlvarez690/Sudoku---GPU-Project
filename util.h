#if !defined(UTIL_H)
#define UTIL_H

#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

inline size_t time_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) == -1) {
    perror("gettimeofday");
    exit(2);
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#endif
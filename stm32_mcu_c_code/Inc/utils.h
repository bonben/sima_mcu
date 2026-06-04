#include <math.h>
#include <stdlib.h>
#include <stddef.h>

float meanError(float* ref, float* res, size_t size);

float maxError(float* ref, float* res, size_t size);

float percMeanError(float* ref, float* res, size_t size);

float percMaxError(float* ref, float* res, size_t size);

void DWT_Init(void);

#include "utils.h"
#include "stm32h7xx.h"
#include <math.h>
#include <stdlib.h>
#include <stddef.h>


float meanError(float* ref, float* res, size_t size){
	float acc = 0;
	for(int i = 0; i < size; i++){
		float err = ref[i] - res[i];
		if(err >= 0) acc += err;
		if(err < 0) acc += -err;
	}
	return acc/size;
}

float maxError(float* ref, float* res, size_t size){
	float max = 0;
	for(int i = 0; i < size; i++){
		float err = ref[i] - res[i];
		if(err > max) max = err;
		if(-err > max) max = -err;
	}
	return max;
}

float percMeanError(float* ref, float* res, size_t size){
	float acc = 0;
	for (size_t i = 0; i < size; i++) {
		if (ref[i] != 0) {
			float percErr = fabsf(ref[i] - res[i]) / fabsf(ref[i]);
			acc += percErr;
		}
	}
	return acc / size;
}

float percMaxError(float* ref, float* res, size_t size) {
	float maxPercErr = 0;
	for (size_t i = 0; i < size; i++) {
		if (ref[i] != 0) {
			float percErr = fabsf(ref[i] - res[i]) / fabsf(ref[i]);
			if (percErr > maxPercErr) {
				maxPercErr = percErr;
			}
		}
	}
	return maxPercErr;
}

void DWT_Init(void) {
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;    // Enable DWT
	DWT->CYCCNT = 0;                                   // Reset counter
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;               // Enable the cycle counter
}

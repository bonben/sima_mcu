#ifndef SIMA_DATA_H
#define SIMA_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

#define SIMA_COUNT 10
#define SIMA_BATCH 1
#define SIMA_LEN 81
#define SIMA_EMBEDDING 32
#define SIMA_HEADS 8
#define SIMA_SAMPLE_SIZE (SIMA_BATCH * SIMA_LEN * SIMA_EMBEDDING)

extern float input[SIMA_COUNT][SIMA_SAMPLE_SIZE];
extern float output[SIMA_COUNT][SIMA_SAMPLE_SIZE];
extern float output_mhsa[SIMA_COUNT][SIMA_SAMPLE_SIZE];

extern float Wq[];
extern int Wq_shape[];
extern float Wk[];
extern int Wk_shape[];
extern float Wv[];
extern int Wv_shape[];
extern float Wo[];
extern int Wo_shape[];

#ifdef __cplusplus
}
#endif

#endif // SIMA_DATA_H

#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "math.h"

void transpose(float* input, size_t rows, size_t cols, float* output);

void softmax_rowwise_float(const float *input, float *output, size_t num_rows, size_t row_size);

void l1_normalize_rows(float* input, size_t out_r, size_t out_c, float* output);

void Mmult(const float* A, const float* B, size_t out_r, size_t out_c, size_t EmbeddedDim, float* Output);

void Mmult_T_QKV(const float* I, const float* Wq, const float* Wk, const float* Wv, size_t out_r, size_t out_c, size_t EmbeddedDim, float* Q_t, float* K_t, float* V_t);

void scaledMmult(const float* A, const float* B, size_t out_r, size_t out_c, size_t EmbeddedDim, float sqrtDim, float* Output);

void Mmultijk(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output);

void Mmultikj(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output);

void Mmultjik(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output);

void Mmultjki(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output);

void Mmultkij(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output);

void Mmultkji(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output);

void Mmultikj_tiled(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output,
			   size_t tile);

void Mmultikj_unrolled(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output);

void attention_engine(const float *query, const float *key, const float *value,
                     float *output,
					 size_t Seq_length, size_t num_keys, size_t EmbeddedDim, size_t PDim, float sqrtDim);

void simpleAttention_engine(const float *query, const float *key, const float *value,
        float *output,
		size_t Seq_length, size_t num_keys, size_t EmbeddedDim, size_t PDim, float sqrtDim);

void multiHeadAttentionEngine(const float* input_data, const float* Wquery, const float* Wkey, const float* Wvalue,
                  float* Wout, float *output,
				  size_t num_heads, size_t seq_length, size_t embed_dim, size_t P, float sqrtDim);

void fusedWeightSelfAttention(const float* input_data, const float* Wquery, const float* Wkey, const float* Wvalue,
                            float* Wout, float *output, size_t num_heads, size_t seq_length, size_t embed_dim, size_t P, float sqrtDim);

void SimMHAttention(const float* input_data, const float* Wquery, const float* Wkey, const float* Wvalue, 
                  float* Wout, float *output,
				  size_t num_heads, size_t seq_length, size_t embed_dim, size_t P, float sqrtDim);




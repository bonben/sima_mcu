#include "Operations.h"
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include "sima_data.h"

#define TILE 1

//----------
void transpose(float* input, size_t rows, size_t cols, float* output){
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			output[j*rows + i] = input[i*cols + j];
		}
	}
}

// Floating point row-wise softmax: input [num_rows x row_size], output [num_rows x row_size]
void softmax_rowwise_float(const float *input, float *output, size_t num_rows, size_t row_size)
{
    for (int row = 0; row < num_rows; ++row)
    {
        // Find max for numerical stability
        float max_val = input[row * row_size];
        for (int col = 1; col < row_size; ++col)
        {
            float v = input[row * row_size + col];
            if (v > max_val)
                max_val = v;
        }

        // Compute exponentials and sum
        float sum = 0.0f;
        for (int col = 0; col < row_size; ++col)
        {
            float exp_val = expf(input[row * row_size + col] - max_val);
            output[row * row_size + col] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for (int col = 0; col < row_size; ++col)
        {
            output[row * row_size + col] /= sum;
        }
    }
}

void lp_normalize_matrix(float *M, int rows, int cols, float p, float eps, float* O) {
    for (int c = 0; c < cols; c++) {
        float norm = 0.0f;

        // compute norm of column c
        if (p == 1.0f) {
            for (int r = 0; r < rows; r++) {
                norm += fabsf(M[r * cols + c]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                norm += powf(fabsf(M[r * cols + c]), p);
            }
            norm = powf(norm, 1.0f / p);
        }

        if (norm < eps) norm = eps;

        // divide each element of column c by its norm
        for (int r = 0; r < rows; r++) {
            O[r * cols + c] = M[r*cols+c]/norm;
        }
    }
}

void Mmult_T_QKV(const float* __restrict I,
                 const float* __restrict Wq,
                 const float* __restrict Wk,
                 const float* __restrict Wv,
                 size_t out_r, size_t out_c, size_t EmbeddedDim,
                 float* __restrict Q_t,
                 float* __restrict K_t,
                 float* __restrict V_t)
{
    for (size_t i = 0; i < out_r; i++) {
        for (size_t j = 0; j < out_c; j++) {

            float accq = 0.0f;
            float acck = 0.0f;
            float accv = 0.0f;

            size_t k = 0;
            size_t baseI = i * EmbeddedDim;

            // Unroll 4 at a time
            for (; k + 3 < EmbeddedDim; k += 4) {
                float a0 = I[baseI + k];
                float a1 = I[baseI + k + 1];
                float a2 = I[baseI + k + 2];
                float a3 = I[baseI + k + 3];

                size_t idx0 = (k)     * out_c + j;
                size_t idx1 = (k + 1) * out_c + j;
                size_t idx2 = (k + 2) * out_c + j;
                size_t idx3 = (k + 3) * out_c + j;

                accq += a0 * Wq[idx0] + a1 * Wq[idx1] + a2 * Wq[idx2] + a3 * Wq[idx3];
                acck += a0 * Wk[idx0] + a1 * Wk[idx1] + a2 * Wk[idx2] + a3 * Wk[idx3];
                accv += a0 * Wv[idx0] + a1 * Wv[idx1] + a2 * Wv[idx2] + a3 * Wv[idx3];
            }

            // Handle remainder
            for (; k < EmbeddedDim; k++) {
                float a = I[baseI + k];
                size_t idx = k * out_c + j;
                accq += a * Wq[idx];
                acck += a * Wk[idx];
                accv += a * Wv[idx];
            }
            int idxO = j * out_r + i;

            Q_t[idxO] = accq;
            K_t[idxO] = acck;
            V_t[idxO] = accv;
        }
    }
}

void Mmult(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output)
{
    // 1) Zero-initialize Output
    for (size_t i = 0; i < out_r * out_c; i++) {
        Output[i] = 0.0f;
    }

    // 2) Outer‐product kernel (i–k–j)
    for (size_t i = 0; i < out_r; i++) {
        const float *rowA = A + i * EmbeddedDim;
        float       *rowO = Output + i * out_c;

        for (size_t k = 0; k < EmbeddedDim; k++) {
            float a = rowA[k];
            const float *rowB = B + k * out_c;

            // 3) Block over j in tiles of TILE
            for (size_t jb = 0; jb < out_c; jb += TILE) {
                size_t jmax = jb + TILE < out_c ? jb + TILE : out_c;

                // 4) Unroll inner loop by 4
                size_t j;

                for (j = jb; j + 3 < jmax; j += 4) {
                    rowO[j + 0] += a * rowB[j + 0];
                    rowO[j + 1] += a * rowB[j + 1];
                    rowO[j + 2] += a * rowB[j + 2];
                    rowO[j + 3] += a * rowB[j + 3];
                }
                // cleanup remainder
                for (; j < jmax; j++) {
                    rowO[j] += a * rowB[j];
                }
            }
        }
    }
}

void Mmultijk(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output)
{
    // 1) Zero-initialize Output
    for (size_t i = 0; i < out_r * out_c; i++) {
        Output[i] = 0.0f;
    }

    for (size_t i = 0; i < out_r; i++) {
        const float *rowA = A + i * EmbeddedDim;
        float       *rowO = Output + i * out_c;
        for (size_t j = 0; j < out_c; j++) {
        	for (size_t k = 0; k < EmbeddedDim; k++){
        		rowO[j] += rowA[k]*B[k*out_c + j];
        	}
        }
    }
}

void Mmultikj(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output)
{
    // 1) Zero-initialize Output
    for (size_t i = 0; i < out_r * out_c; i++) {
        Output[i] = 0.0f;
    }

    for (size_t i = 0; i < out_r; i++) {
        const float *rowA = A + i * EmbeddedDim;
        float       *rowO = Output + i * out_c;
        for (size_t k = 0; k < EmbeddedDim; k++){
        	float a = rowA[k];
        	for (size_t j = 0; j < out_c; j++){
        		rowO[j] += a*B[k*out_c + j];
        	}
        }
    }
}

void Mmultjik(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output)
{
    // 1) Zero-initialize Output
    for (size_t i = 0; i < out_r * out_c; i++) {
        Output[i] = 0.0f;
    }

    for (size_t j = 0; j < out_c; j++){
        for (size_t i = 0; i < out_r; i++) {
            const float *rowA = A + i * EmbeddedDim;
            float       *posO = Output + i * out_c + j;
        	for (size_t k = 0; k < EmbeddedDim; k++){
        		*posO += rowA[k]*B[k*out_c + j];
        	}
        }
    }
}

void Mmultjki(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output)
{
    // 1) Zero-initialize Output
    for (size_t i = 0; i < out_r * out_c; i++) {
        Output[i] = 0.0f;
    }

    for (size_t j = 0; j < out_c; j++){
        for (size_t k = 0; k < EmbeddedDim; k++){
        	float b = B[k*out_c + j];
        	for (size_t i = 0; i < out_r; i++) {
        		Output[i * out_c + j] += A[i * EmbeddedDim + k] * b;
        	}
        }
    }
}

void Mmultkij(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output)
{
    // 1) Zero-initialize Output
    for (size_t i = 0; i < out_r * out_c; i++) {
        Output[i] = 0.0f;
    }

    for (size_t k = 0; k < EmbeddedDim; k++){
    	float* rowB = B + k*out_c;
        for (size_t i = 0; i < out_r; i++){
        	float* rowO = Output + i*out_c;
        	for (size_t j = 0; j < out_c; j++) {
        		rowO[j] += A[i * EmbeddedDim + k] * rowB[j];
        	}
        }
    }
}

void Mmultkji(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output)
{
    // 1) Zero-initialize Output
    for (size_t i = 0; i < out_r * out_c; i++) {
        Output[i] = 0.0f;
    }

    for (size_t k = 0; k < EmbeddedDim; k++){
    	float* rowB = B + k*out_c;
        for (size_t j = 0; j < out_c; j++){
        	for (size_t i = 0; i < out_r; i++) {
        		Output[i*out_c+j] += A[i * EmbeddedDim + k] * rowB[j];
        	}
        }
    }
}

void Mmultikj_tiled(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output,
			   size_t tile)
{
    // Initialize C
    for (size_t i = 0; i < out_r * out_c; i++)
    	Output[i] = 0.0f;

    // Tiled i-k-j scheduling
    for (size_t ii = 0; ii < out_r; ii += tile) {
        size_t i_max = (ii + tile > out_r) ? out_r : ii + tile;

        for (size_t kk = 0; kk < EmbeddedDim; kk += tile) {
            size_t k_max = (kk + tile > EmbeddedDim) ? EmbeddedDim : kk + tile;

            for (size_t jj = 0; jj < out_c; jj += tile) {
                size_t j_max = (jj + tile > out_c) ? out_c : jj + tile;

                // Now compute the small block
                for (size_t i = ii; i < i_max; i++) {
                	float* rowO = Output + i*out_c;
                    for (size_t k = kk; k < k_max; k++) {
                        float a_ik = A[i * EmbeddedDim + k]; // row-major
                        float* rowB = B + k * out_c;
                        for (size_t j = jj; j < j_max; j++) {
                        	rowO[j] += a_ik * rowB[j];
                        }
                    }
                }
            }
        }
    }
}

void Mmultikj_unrolled(const float * __restrict A,
               const float * __restrict B,
               size_t out_r,
               size_t out_c,
               size_t EmbeddedDim,
               float * __restrict Output)
{
    // Initialize C
    for (size_t i = 0; i < out_r * out_c; i++)
    	Output[i] = 0.0f;

    // i-k-j order with unrolling
    for (size_t i = 0; i < out_r; i++) {
    	float* rowO = Output + i * out_c;
        for (size_t k = 0; k < EmbeddedDim; k++) {
            float a_ik = A[i * EmbeddedDim + k];  // reuse this value
            float* rowB = B + k * out_c;
            size_t j = 0;

            // Unroll j loop by 4
            for (; j + 1 < out_c; j += 2) {
            	rowO[j]     += a_ik * rowB[j];
            	rowO[j + 1] += a_ik * rowB[j + 1];
            }

            // Handle leftover columns
            for (; j < out_c; j++) {
            	rowO[j] += a_ik * rowB[j];
            }
        }
    }
}

/*
 * Returns A*B_t/sqrt(embedded_dim)
 *
 * */
void scaledMmult(const float* A, const float* B, size_t out_r, size_t out_c, size_t EmbeddedDim, float sqrtDim, float* Output){
	for(size_t i = 0; i < out_r; i++){
		for(size_t j = 0; j < out_c; j++){
			float acc = 0.0f;
			for(size_t k = 0; k < EmbeddedDim; k++){
				acc += A[i*EmbeddedDim + k]*B[j*EmbeddedDim + k];
			}
			Output[i*out_c + j] = acc/sqrtDim;
		}
	}
}

void attention_engine(const float *query, const float *key, const float *value,
                    float *output,
					 size_t Seq_length, size_t num_keys, size_t EmbeddedDim, size_t PDim, float sqrtDim)
{
   // float *scores = (float*)malloc(Seq_length * num_keys * sizeof(float));
   float scores[SIMA_LEN * SIMA_LEN];
   // float *softmax_scores = (float*)malloc(Seq_length * num_keys * sizeof(float));
   float softmax_scores[SIMA_LEN * SIMA_LEN];
   scaledMmult(query, key, Seq_length, num_keys, EmbeddedDim, sqrtDim, scores);

   softmax_rowwise_float(scores, softmax_scores, Seq_length, num_keys);

   Mmult(softmax_scores, value, Seq_length, PDim, Seq_length, output);
}

void simpleAttention_engine(const float *query, const float *key, const float *value,
        float *output,
		size_t Seq_length, size_t num_keys, size_t EmbeddedDim, size_t PDim, float sqrtDim)
{
	float q[SIMA_LEN * (SIMA_EMBEDDING/SIMA_HEADS)];
	float k[SIMA_LEN * (SIMA_EMBEDDING/SIMA_HEADS)];
	float k_t[SIMA_LEN * (SIMA_EMBEDDING/SIMA_HEADS)];
	lp_normalize_matrix(query, Seq_length, EmbeddedDim, 1.0, 0.000001, q);
	lp_normalize_matrix(key, Seq_length, EmbeddedDim, 1.0, 0.000001, k);
	transpose(k, Seq_length, EmbeddedDim, k_t); //transpose key

	if(Seq_length <= EmbeddedDim){
		float scores[SIMA_LEN * SIMA_LEN];
		Mmultikj(q, k_t, Seq_length, Seq_length, EmbeddedDim, scores);
		Mmultikj(scores, value, Seq_length, EmbeddedDim, Seq_length, output);
	} else {
		float scores[(SIMA_EMBEDDING/SIMA_HEADS) * (SIMA_EMBEDDING/SIMA_HEADS)];
		Mmultikj(k_t, value, EmbeddedDim, EmbeddedDim, Seq_length, scores);
		Mmultikj(q, scores, Seq_length, EmbeddedDim, EmbeddedDim, output);
	}
}

// Multi-head attention engine (single batch, float, no masking)
// query, key, value: [num_heads * seq_length * embed_dim]
// output: [num_heads * seq_length * embed_dim]
// num_heads: number of attention heads
// seq_length: sequence length (number of queries/keys/values per head)
// embed_dim: embedding dimension per head
void multiHeadAttentionEngine(const float* input_data, const float* Wquery, const float* Wkey, const float* Wvalue,
                            float* Wout, float *output, size_t num_heads, size_t seq_length, size_t embed_dim, size_t P, float sqrtDim)
{
	size_t embed_dim_head = P/num_heads;
    float Q[SIMA_LEN * SIMA_EMBEDDING];
	float K[SIMA_LEN * SIMA_EMBEDDING];
	float V[SIMA_LEN * SIMA_EMBEDDING];
	float Y[SIMA_LEN * SIMA_EMBEDDING];
	float rY[SIMA_LEN * SIMA_EMBEDDING];

	Mmult_T_QKV(input_data, Wquery, Wkey, Wvalue, seq_length, P, embed_dim, Q, K, V);

   // For each head, run the attention engine
	float q_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float k_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float v_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float y_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
   for (size_t h = 0; h < num_heads; ++h)
   {
       transpose(Q + h * seq_length * embed_dim_head, embed_dim_head, seq_length, q_head);
       transpose(K + h * seq_length * embed_dim_head, embed_dim_head, seq_length, k_head);
       transpose(V + h * seq_length * embed_dim_head, embed_dim_head, seq_length, v_head);

       attention_engine(q_head, k_head, v_head, y_head, seq_length, seq_length, embed_dim_head, embed_dim_head, sqrtDim);

       transpose(y_head, seq_length, embed_dim_head, Y + h * seq_length * embed_dim_head);


   }
   transpose(Y, P, seq_length, rY);
   Mmultikj(rY, Wout, seq_length, embed_dim, P, output);

}

void fusedWeightSelfAttention(const float* input_data, const float* Wquery, const float* Wkey, const float* Wvalue,
                            float* Wout, float *output, size_t num_heads, size_t seq_length, size_t embed_dim, size_t P, float sqrtDim)
{
	size_t embed_dim_head = P/num_heads;
	float W[SIMA_EMBEDDING * SIMA_EMBEDDING];
    float M[SIMA_LEN * SIMA_EMBEDDING];
	float It[SIMA_EMBEDDING * SIMA_LEN];
	float V[SIMA_LEN * SIMA_EMBEDDING];
	float Y[SIMA_LEN * SIMA_EMBEDDING];
	float rY[SIMA_LEN * SIMA_EMBEDDING];
    float Wkey_t[SIMA_EMBEDDING * SIMA_EMBEDDING];
    transpose(Wkey, Wkey_t, SIMA_EMBEDDING, SIMA_EMBEDDING);

	Mmultikj(Wquery, Wkey_t, SIMA_EMBEDDING, SIMA_EMBEDDING, SIMA_EMBEDDING, W);
    transpose(input_data, It, SIMA_LEN, SIMA_EMBEDDING);

	Mmultikj(input_data, W, SIMA_LEN, SIMA_EMBEDDING, SIMA_EMBEDDING, M);

   // For each head, run the attention engine
	float m_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float i_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float v_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float y_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
   for (size_t h = 0; h < num_heads; ++h)
   {
       transpose(M + h * seq_length * embed_dim_head, embed_dim_head, seq_length, m_head);
       transpose(It + h * seq_length * embed_dim_head, embed_dim_head, seq_length, i_head);
       transpose(V + h * seq_length * embed_dim_head, embed_dim_head, seq_length, v_head);

       attention_engine(m_head, i_head, v_head, y_head, seq_length, seq_length, embed_dim_head, embed_dim_head, sqrtDim);

       transpose(y_head, seq_length, embed_dim_head, Y + h * seq_length * embed_dim_head);


   }
   transpose(Y, P, seq_length, rY);
   Mmultikj(rY, Wout, seq_length, embed_dim, P, output);

}

void SimMHAttention(const float* input_data, const float* Wquery, const float* Wkey, const float* Wvalue,
                  float* Wout, float* output,
				  size_t num_heads, size_t seq_length, size_t embed_dim, size_t P, float sqrtDim)
{
	float Q[SIMA_LEN * SIMA_EMBEDDING];
	float K[SIMA_LEN * SIMA_EMBEDDING];
	float V[SIMA_LEN * SIMA_EMBEDDING];
	float Y[SIMA_LEN * SIMA_EMBEDDING];
	float rY[SIMA_LEN * SIMA_EMBEDDING];

	Mmult_T_QKV(input_data, Wquery, Wkey, Wvalue, SIMA_LEN, SIMA_EMBEDDING, SIMA_EMBEDDING, Q, K, V);

	float q_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float k_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float v_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	float y_head[SIMA_LEN * SIMA_EMBEDDING/SIMA_HEADS];
	size_t embed_dim_head = P/num_heads;
	for (size_t h = 0; h < num_heads; ++h)
	{
		transpose(Q + h * SIMA_LEN * embed_dim_head, embed_dim_head, SIMA_LEN, q_head);
		transpose(K + h * SIMA_LEN * embed_dim_head, embed_dim_head, SIMA_LEN, k_head);
		transpose(V + h * SIMA_LEN * embed_dim_head, embed_dim_head, SIMA_LEN, v_head);

		simpleAttention_engine(q_head, k_head, v_head, y_head, SIMA_LEN, SIMA_LEN, embed_dim_head, embed_dim_head, sqrtDim);

        transpose(y_head, SIMA_LEN, embed_dim_head, Y + h * SIMA_LEN * embed_dim_head);
	}
    transpose(Y, SIMA_EMBEDDING, SIMA_LEN, rY);
    Mmultikj(rY, Wout, SIMA_LEN, SIMA_EMBEDDING, SIMA_EMBEDDING, output);
}


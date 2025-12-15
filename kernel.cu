#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// DCT-BASED WATERMARKING (JPEG-Resistant)
// ============================================================================

#define BLOCK_SIZE 8
#define PI 3.14159265358979323846

// 2D DCT kernel
__global__ void dct2d_kernel(float* img, float* dct_out, int width, int height) {
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (bx + tx >= width || by + ty >= height) return;

    __shared__ float block[BLOCK_SIZE][BLOCK_SIZE];

    // Load block into shared memory
    block[ty][tx] = img[(by + ty) * width + (bx + tx)];
    __syncthreads();

    // Compute DCT coefficient
    float sum = 0.0f;
    for (int y = 0; y < BLOCK_SIZE; y++) {
        for (int x = 0; x < BLOCK_SIZE; x++) {
            float cu = (tx == 0) ? 1.0f/sqrtf(2.0f) : 1.0f;
            float cv = (ty == 0) ? 1.0f/sqrtf(2.0f) : 1.0f;
            sum += block[y][x] *
                   cosf((2*x + 1) * tx * PI / (2.0f * BLOCK_SIZE)) *
                   cosf((2*y + 1) * ty * PI / (2.0f * BLOCK_SIZE)) *
                   cu * cv;
        }
    }

    dct_out[(by + ty) * width + (bx + tx)] = sum * 0.25f;
}

// Inverse DCT kernel
__global__ void idct2d_kernel(float* dct_in, float* img_out, int width, int height) {
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (bx + tx >= width || by + ty >= height) return;

    __shared__ float dct_block[BLOCK_SIZE][BLOCK_SIZE];

    dct_block[ty][tx] = dct_in[(by + ty) * width + (bx + tx)];
    __syncthreads();

    float sum = 0.0f;
    for (int v = 0; v < BLOCK_SIZE; v++) {
        for (int u = 0; u < BLOCK_SIZE; u++) {
            float cu = (u == 0) ? 1.0f/sqrtf(2.0f) : 1.0f;
            float cv = (v == 0) ? 1.0f/sqrtf(2.0f) : 1.0f;
            sum += dct_block[v][u] * cu * cv *
                   cosf((2*tx + 1) * u * PI / (2.0f * BLOCK_SIZE)) *
                   cosf((2*ty + 1) * v * PI / (2.0f * BLOCK_SIZE));
        }
    }

    img_out[(by + ty) * width + (bx + tx)] = sum * 0.25f;
}

// Embed watermark in mid-frequency DCT coefficients
__global__ void embed_dct_watermark(float* dct, int width, int height, int key, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = (width / BLOCK_SIZE) * (height / BLOCK_SIZE);

    if (idx >= total_blocks) return;

    int block_y = idx / (width / BLOCK_SIZE);
    int block_x = idx % (width / BLOCK_SIZE);

    // Embed in mid-frequency coefficients (3,3) and (4,4)
    int base_y = block_y * BLOCK_SIZE;
    int base_x = block_x * BLOCK_SIZE;

    float watermark_bit = (key % 2 != 0) ? 1.0f : -1.0f;

    // Modify mid-frequency coefficients
    dct[(base_y + 3) * width + (base_x + 3)] += alpha * watermark_bit;
    dct[(base_y + 4) * width + (base_x + 4)] += alpha * watermark_bit * 0.5f;
}

// Extract watermark from DCT
__global__ void extract_dct_watermark(float* dct_marked, float* dct_original,
                                       int width, int height, int* d_sum, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = (width / BLOCK_SIZE) * (height / BLOCK_SIZE);

    if (idx >= total_blocks) return;

    int block_y = idx / (width / BLOCK_SIZE);
    int block_x = idx % (width / BLOCK_SIZE);
    int base_y = block_y * BLOCK_SIZE;
    int base_x = block_x * BLOCK_SIZE;

    float diff = dct_marked[(base_y + 3) * width + (base_x + 3)] -
                 dct_original[(base_y + 3) * width + (base_x + 3)];

    int bit = (diff > 0) ? 1 : 0;
    atomicAdd(d_sum, bit);
}

// ============================================================================
// DWT-BASED WATERMARKING
// ============================================================================

__device__ void haar_1d_forward(float* data, int n, float* temp) {
    int h = n >> 1;
    for (int i = 0; i < h; i++) {
        temp[i] = (data[2*i] + data[2*i+1]) * 0.707107f;
        temp[h+i] = (data[2*i] - data[2*i+1]) * 0.707107f;
    }
    for (int i = 0; i < n; i++) data[i] = temp[i];
}

__device__ void haar_1d_inverse(float* data, int n, float* temp) {
    int h = n >> 1;
    for (int i = 0; i < h; i++) {
        temp[2*i] = (data[i] + data[h+i]) * 0.707107f;
        temp[2*i+1] = (data[i] - data[h+i]) * 0.707107f;
    }
    for (int i = 0; i < n; i++) data[i] = temp[i];
}

__global__ void dwt2d_forward(float* img, int width, int height) {
    extern __shared__ float shared_mem[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height) {
        haar_1d_forward(&img[row * width], width, shared_mem);
    }
}

__global__ void dwt2d_inverse(float* img, int width, int height) {
    extern __shared__ float shared_mem[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height) {
        haar_1d_inverse(&img[row * width], width, shared_mem);
    }
}

// ============================================================================
// LSB WATERMARKING (Simple)
// ============================================================================

__global__ void lsb_embed(unsigned char* img, int size, int key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        img[idx] = (img[idx] & 0xFE) | ((key % 2 != 0) ? 1 : 0);
    }
}

__global__ void lsb_extract(unsigned char* img, int size, int* sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(sum, img[idx] & 0x01);
    }
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

__global__ void uint8_to_float(unsigned char* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = (float)in[idx];
}

__global__ void float_to_uint8(float* in, unsigned char* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = fmaxf(0.0f, fminf(255.0f, in[idx]));
        out[idx] = (unsigned char)val;
    }
}

// Calculate PSNR
__global__ void compute_mse(unsigned char* img1, unsigned char* img2, int size, float* mse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = (float)img1[idx] - (float)img2[idx];
        atomicAdd(mse, diff * diff);
    }
}

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

extern "C" {

// DCT-based watermarking
void launch_dct_watermark(unsigned char* h_img, int width, int height, int key, float alpha) {
    unsigned char *d_img_u8;
    float *d_img_f, *d_dct;
    size_t size_u8 = width * height;
    size_t size_f = size_u8 * sizeof(float);

    cudaMalloc(&d_img_u8, size_u8);
    cudaMalloc(&d_img_f, size_f);
    cudaMalloc(&d_dct, size_f);

    cudaMemcpy(d_img_u8, h_img, size_u8, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size_u8 + threads - 1) / threads;
    uint8_to_float<<<blocks, threads>>>(d_img_u8, d_img_f, size_u8);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dct2d_kernel<<<gridDim, blockDim>>>(d_img_f, d_dct, width, height);

    int total_blocks = (width / BLOCK_SIZE) * (height / BLOCK_SIZE);
    int embed_blocks = (total_blocks + 255) / 256;
    embed_dct_watermark<<<embed_blocks, 256>>>(d_dct, width, height, key, alpha);

    idct2d_kernel<<<gridDim, blockDim>>>(d_dct, d_img_f, width, height);
    float_to_uint8<<<blocks, threads>>>(d_img_f, d_img_u8, size_u8);

    cudaDeviceSynchronize();
    cudaMemcpy(h_img, d_img_u8, size_u8, cudaMemcpyDeviceToHost);

    cudaFree(d_img_u8);
    cudaFree(d_img_f);
    cudaFree(d_dct);
}

// LSB watermarking (with CPU fallback)
void launch_lsb_watermark(unsigned char* h_img, int width, int height, int key) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        // CPU fallback
        for (int i = 0; i < width * height; i++) {
            h_img[i] = (h_img[i] & 0xFE) | ((key % 2 != 0) ? 1 : 0);
        }
        return;
    }

    unsigned char* d_img;
    size_t size = width * height;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    lsb_embed<<<blocks, threads>>>(d_img, size, key);

    cudaDeviceSynchronize();
    cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
}

// LSB extraction
int launch_lsb_extract(unsigned char* h_img, int width, int height) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        int sum = 0;
        for (int i = 0; i < width * height; i++) {
            sum += h_img[i] & 0x01;
        }
        return (sum > (width * height / 2)) ? 1 : 0;
    }

    unsigned char* d_img;
    int *d_sum, h_sum = 0;
    size_t size = width * height;

    cudaMalloc(&d_img, size);
    cudaMalloc(&d_sum, sizeof(int));
    cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(int));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    lsb_extract<<<blocks, threads>>>(d_img, size, d_sum);

    cudaDeviceSynchronize();
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_sum);

    return (h_sum > (width * height / 2)) ? 1 : 0;
}

// Compute PSNR between two images
float compute_psnr(unsigned char* img1, unsigned char* img2, int width, int height) {
    unsigned char *d_img1, *d_img2;
    float *d_mse, h_mse = 0.0f;
    size_t size = width * height;

    cudaMalloc(&d_img1, size);
    cudaMalloc(&d_img2, size);
    cudaMalloc(&d_mse, sizeof(float));

    cudaMemcpy(d_img1, img1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2, size, cudaMemcpyHostToDevice);
    cudaMemset(d_mse, 0, sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_mse<<<blocks, threads>>>(d_img1, d_img2, size, d_mse);

    cudaDeviceSynchronize();
    cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_mse);

    h_mse /= size;
    if (h_mse == 0) return 100.0f;  // Identical images
    return 10.0f * log10f(255.0f * 255.0f / h_mse);
}

} // extern "C"

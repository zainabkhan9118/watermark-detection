#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// DWT-BASED WATERMARKING KERNELS
// ============================================================================

// Haar wavelet transform (1D, in-place)
__device__ void haar_1d_forward(float* data, int n) {
    float* temp = new float[n];
    int h = n >> 1;
    
    for (int i = 0; i < h; i++) {
        temp[i] = (data[2*i] + data[2*i+1]) * 0.5f;           // Approximation
        temp[h+i] = (data[2*i] - data[2*i+1]) * 0.5f;         // Detail
    }
    
    for (int i = 0; i < n; i++) {
        data[i] = temp[i];
    }
    delete[] temp;
}

__device__ void haar_1d_inverse(float* data, int n) {
    float* temp = new float[n];
    int h = n >> 1;
    
    for (int i = 0; i < h; i++) {
        temp[2*i] = data[i] + data[h+i];       // Reconstruct even
        temp[2*i+1] = data[i] - data[h+i];     // Reconstruct odd
    }
    
    for (int i = 0; i < n; i++) {
        data[i] = temp[i];
    }
    delete[] temp;
}

// 2D DWT Forward Transform Kernel
__global__ void dwt2d_forward_kernel(float* img, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height) {
        // Transform rows
        haar_1d_forward(&img[row * width], width);
    }
    __syncthreads();
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        // Transform columns
        float* column = new float[height];
        for (int i = 0; i < height; i++) {
            column[i] = img[i * width + col];
        }
        haar_1d_forward(column, height);
        for (int i = 0; i < height; i++) {
            img[i * width + col] = column[i];
        }
        delete[] column;
    }
}

// 2D DWT Inverse Transform Kernel
__global__ void dwt2d_inverse_kernel(float* img, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < width) {
        // Inverse transform columns
        float* column = new float[height];
        for (int i = 0; i < height; i++) {
            column[i] = img[i * width + col];
        }
        haar_1d_inverse(column, height);
        for (int i = 0; i < height; i++) {
            img[i * width + col] = column[i];
        }
        delete[] column;
    }
    __syncthreads();
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height) {
        // Inverse transform rows
        haar_1d_inverse(&img[row * width], width);
    }
}

// Embed watermark in DWT coefficients
__global__ void embed_watermark_dwt(float* dwt_coeffs, int width, int height, int key, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (width * height) / 4;  // Only modify LL band (top-left quadrant)
    
    if (idx < total) {
        int row = idx / (width / 2);
        int col = idx % (width / 2);
        int pos = row * width + col;
        
        // Embed watermark bit in LL coefficients
        float watermark_bit = (key % 2 != 0) ? 1.0f : -1.0f;
        dwt_coeffs[pos] += alpha * watermark_bit;
    }
}

// Extract watermark from DWT coefficients
__global__ void extract_watermark_dwt(float* dwt_coeffs, int width, int height, float* original_coeffs, int* d_sum, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (width * height) / 4;
    
    if (idx < total) {
        int row = idx / (width / 2);
        int col = idx % (width / 2);
        int pos = row * width + col;
        
        // Extract watermark bit
        float diff = dwt_coeffs[pos] - original_coeffs[pos];
        int bit = (diff > 0) ? 1 : 0;
        atomicAdd(d_sum, bit);
    }
}

// Convert uint8 to float
__global__ void uint8_to_float(unsigned char* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (float)in[idx];
    }
}

// Convert float to uint8 with clamping
__global__ void float_to_uint8(float* in, unsigned char* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = in[idx];
        val = fmaxf(0.0f, fminf(255.0f, val));
        out[idx] = (unsigned char)val;
    }
}

// ============================================================================
// SIMPLE LSB WATERMARKING (FALLBACK)
// ============================================================================

__global__ void watermarkKernel(unsigned char* d_img, int width, int height, int key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        unsigned char pixel = d_img[idx];
        pixel = pixel & 0xFE;
        if (key % 2 != 0) {
            pixel = pixel | 0x01;
        }
        d_img[idx] = pixel;
    }
}

__global__ void extractKernel(unsigned char* d_img, int width, int height, int* d_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int lsb = d_img[idx] & 0x01;
        atomicAdd(d_sum, lsb);
    }
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

extern "C" {
    // DWT-based watermarking
    void launch_dwt_watermark_kernel(unsigned char* h_img, int width, int height, int key) {
        unsigned char* d_img_uint8;
        float* d_img_float;
        size_t size_uint8 = width * height * sizeof(unsigned char);
        size_t size_float = width * height * sizeof(float);
        cudaError_t err;

        // Allocate memory
        cudaMalloc(&d_img_uint8, size_uint8);
        cudaMalloc(&d_img_float, size_float);
        
        // Copy to device
        cudaMemcpy(d_img_uint8, h_img, size_uint8, cudaMemcpyHostToDevice);
        
        // Convert to float
        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;
        uint8_to_float<<<blocks, threads>>>(d_img_uint8, d_img_float, width * height);
        
        // Apply DWT
        dim3 blockDim(16, 16);
        dim3 gridDim((width + 15) / 16, (height + 15) / 16);
        dwt2d_forward_kernel<<<gridDim, blockDim>>>(d_img_float, width, height);
        
        // Embed watermark
        float alpha = 10.0f;  // Embedding strength
        embed_watermark_dwt<<<blocks, threads>>>(d_img_float, width, height, key, alpha);
        
        // Apply inverse DWT
        dwt2d_inverse_kernel<<<gridDim, blockDim>>>(d_img_float, width, height);
        
        // Convert back to uint8
        float_to_uint8<<<blocks, threads>>>(d_img_float, d_img_uint8, width * height);
        
        cudaDeviceSynchronize();
        
        // Copy back
        cudaMemcpy(h_img, d_img_uint8, size_uint8, cudaMemcpyDeviceToHost);
        
        cudaFree(d_img_uint8);
        cudaFree(d_img_float);
    }

    // Simple LSB watermarking with CPU fallback
    void launch_watermark_kernel(unsigned char* h_img, int width, int height, int key) {
        cudaError_t err;
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        
        if (deviceCount == 0) {
            // CPU fallback
            for (int i = 0; i < width * height; i++) {
                unsigned char pixel = h_img[i];
                pixel = pixel & 0xFE;
                if (key % 2 != 0) {
                    pixel = pixel | 0x01;
                }
                h_img[i] = pixel;
            }
            return;
        }

        unsigned char* d_img;
        size_t size = width * height * sizeof(unsigned char);
        
        err = cudaMalloc(&d_img, size);
        if (err != cudaSuccess) return;
        
        cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);
        
        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;
        watermarkKernel<<<blocks, threads>>>(d_img, width, height, key);
        
        cudaDeviceSynchronize();
        cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);
        cudaFree(d_img);
    }

    int launch_extract_kernel(unsigned char* h_img, int width, int height) {
        cudaError_t err;
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        
        if (deviceCount == 0) {
            // CPU fallback
            int h_sum = 0;
            for (int i = 0; i < width * height; i++) {
                h_sum += (h_img[i] & 0x01);
            }
            return (h_sum > (width * height / 2)) ? 1 : 0;
        }

        unsigned char* d_img;
        int* d_sum;
        int h_sum = 0;
        size_t size = width * height * sizeof(unsigned char);
        
        cudaMalloc(&d_img, size);
        cudaMalloc(&d_sum, sizeof(int));
        
        cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);
        cudaMemset(d_sum, 0, sizeof(int));
        
        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;
        extractKernel<<<blocks, threads>>>(d_img, width, height, d_sum);
        
        cudaDeviceSynchronize();
        cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_img);
        cudaFree(d_sum);
        
        return (h_sum > (width * height / 2)) ? 1 : 0;
    }
}

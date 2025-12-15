#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

// Defined in kernel.cu
void launch_dct_watermark(unsigned char* h_img, int width, int height, int key, float alpha);
void launch_lsb_watermark(unsigned char* h_img, int width, int height, int key);
int launch_lsb_extract(unsigned char* h_img, int width, int height);
float compute_psnr(unsigned char* img1, unsigned char* img2, int width, int height);

#define WIDTH 1024
#define HEIGHT 1024

// Read PGM image
unsigned char* read_pgm(const char* filename, int* w, int* h) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;

    char buf[16];
    if (!fgets(buf, sizeof(buf), fp)) return NULL;
    if (buf[0] != 'P' || buf[1] != '5') return NULL;

    int c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n');
        c = getc(fp);
    }
    ungetc(c, fp);

    if (fscanf(fp, "%d %d", w, h) != 2) return NULL;
    int maxval;
    if (fscanf(fp, "%d", &maxval) != 1) return NULL;
    fgetc(fp);

    unsigned char* data = (unsigned char*)malloc(*w * *h);
    fread(data, 1, *w * *h, fp);
    fclose(fp);
    return data;
}

void write_pgm(const char* filename, unsigned char* data, int w, int h) {
    FILE* fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%d %d\n255\n", w, h);
    fwrite(data, 1, w * h, fp);
    fclose(fp);
}

// Attack simulations
void add_gaussian_noise(unsigned char* img, int size, float sigma) {
    for (int i = 0; i < size; i++) {
        float noise = ((float)rand() / RAND_MAX - 0.5f) * sigma;
        int val = (int)img[i] + (int)noise;
        img[i] = (unsigned char)(val < 0 ? 0 : (val > 255 ? 255 : val));
    }
}

void jpeg_simulate(unsigned char* img, int size, int quality) {
    // Simulate JPEG compression by quantization
    int q = (100 - quality) / 10 + 1;
    for (int i = 0; i < size; i++) {
        img[i] = (img[i] / q) * q;
    }
}

void salt_pepper_noise(unsigned char* img, int size, float prob) {
    for (int i = 0; i < size; i++) {
        float r = (float)rand() / RAND_MAX;
        if (r < prob / 2) img[i] = 0;
        else if (r < prob) img[i] = 255;
    }
}

int main() {
    const char* input_dir = "input_images";
    const char* output_dir = "output_images";
    const char* attacked_dir = "attacked_images";
    int num_images = 20;

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  ADVANCED WATERMARKING SYSTEM - COMPREHENSIVE EVALUATION\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\n");

    // Statistics
    float total_psnr_lsb = 0.0f, total_psnr_dct = 0.0f;
    int lsb_correct = 0, dct_correct = 0;
    int lsb_robust_gaussian = 0, dct_robust_gaussian = 0;
    int lsb_robust_jpeg = 0, dct_robust_jpeg = 0;

    double start_time = omp_get_wtime();

    printf("Phase 1: Embedding Watermarks...\n");
    printf("────────────────────────────────────────────────────────────\n");

    #pragma omp parallel for num_threads(4) reduction(+:total_psnr_lsb,total_psnr_dct,lsb_correct,dct_correct)
    for (int i = 0; i < num_images; i++) {
        char in_path[256], out_lsb[256], out_dct[256];
        sprintf(in_path, "%s/img_%03d.pgm", input_dir, i);
        sprintf(out_lsb, "%s/img_%03d_lsb.pgm", output_dir, i);
        sprintf(out_dct, "%s/img_%03d_dct.pgm", output_dir, i);

        int w, h;
        unsigned char* original = read_pgm(in_path, &w, &h);
        if (!original) continue;

        // Create copies for different methods
        unsigned char* img_lsb = (unsigned char*)malloc(w * h);
        unsigned char* img_dct = (unsigned char*)malloc(w * h);
        memcpy(img_lsb, original, w * h);
        memcpy(img_dct, original, w * h);

        int secret_key = 12345;

        // LSB watermarking
        launch_lsb_watermark(img_lsb, w, h, secret_key);
        write_pgm(out_lsb, img_lsb, w, h);
        float psnr_lsb = compute_psnr(original, img_lsb, w, h);
        total_psnr_lsb += psnr_lsb;

        // DCT watermarking
        launch_dct_watermark(img_dct, w, h, secret_key, 15.0f);
        write_pgm(out_dct, img_dct, w, h);
        float psnr_dct = compute_psnr(original, img_dct, w, h);
        total_psnr_dct += psnr_dct;

        // Verify extraction
        int extracted_lsb = launch_lsb_extract(img_lsb, w, h);
        int expected = (secret_key % 2 != 0) ? 1 : 0;
        if (extracted_lsb == expected) lsb_correct++;

        #pragma omp critical
        {
            printf("  [%02d] LSB: PSNR=%.2f dB | DCT: PSNR=%.2f dB\n",
                   i, psnr_lsb, psnr_dct);
        }

        free(original);
        free(img_lsb);
        free(img_dct);
    }

    printf("\n");
    printf("Phase 2: Robustness Testing...\n");
    printf("────────────────────────────────────────────────────────────\n");

    // Test first 5 images with attacks
    for (int i = 0; i < 5; i++) {
        char lsb_path[256], dct_path[256];
        sprintf(lsb_path, "%s/img_%03d_lsb.pgm", output_dir, i);
        sprintf(dct_path, "%s/img_%03d_dct.pgm", output_dir, i);

        int w, h;
        unsigned char* img_lsb = read_pgm(lsb_path, &w, &h);
        unsigned char* img_dct = read_pgm(dct_path, &w, &h);

        if (!img_lsb || !img_dct) continue;

        // Test 1: Gaussian Noise
        unsigned char* test_lsb = (unsigned char*)malloc(w * h);
        unsigned char* test_dct = (unsigned char*)malloc(w * h);
        memcpy(test_lsb, img_lsb, w * h);
        memcpy(test_dct, img_dct, w * h);

        add_gaussian_noise(test_lsb, w * h, 10.0f);
        add_gaussian_noise(test_dct, w * h, 10.0f);

        int ext_lsb = launch_lsb_extract(test_lsb, w, h);
        int expected = 1;
        if (ext_lsb == expected) lsb_robust_gaussian++;

        printf("  [%02d] Gaussian Noise: LSB=%s | DCT=Pending\n",
               i, (ext_lsb == expected) ? "✓" : "✗");

        // Test 2: JPEG Simulation
        memcpy(test_lsb, img_lsb, w * h);
        memcpy(test_dct, img_dct, w * h);

        jpeg_simulate(test_lsb, w * h, 75);
        jpeg_simulate(test_dct, w * h, 75);

        ext_lsb = launch_lsb_extract(test_lsb, w, h);
        if (ext_lsb == expected) lsb_robust_jpeg++;

        printf("  [%02d] JPEG Q=75:       LSB=%s | DCT=Pending\n",
               i, (ext_lsb == expected) ? "✓" : "✗");

        free(test_lsb);
        free(test_dct);
        free(img_lsb);
        free(img_dct);
    }

    double end_time = omp_get_wtime();

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  FINAL RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("Performance Metrics:\n");
    printf("  Total Processing Time: %.4f seconds\n", end_time - start_time);
    printf("  Images Processed: %d\n", num_images);
    printf("  Avg Time per Image: %.4f seconds\n", (end_time - start_time) / num_images);
    printf("\n");
    printf("Quality Metrics (PSNR):  \n");
    printf("  LSB Method: %.2f dB (avg)\n", total_psnr_lsb / num_images);
    printf("  DCT Method: %.2f dB (avg)\n", total_psnr_dct / num_images);
    printf("\n");
    printf("Extraction Accuracy:\n");
    printf("  LSB: %d/%d (%.1f%%)\n", lsb_correct, num_images, 100.0f * lsb_correct / num_images);
    printf("  DCT: Pending implementation\n");
    printf("\n");
    printf("Robustness (5 test images):\n");
    printf("  Gaussian Noise: LSB %d/5 | DCT Pending\n", lsb_robust_gaussian);
    printf("  JPEG Q=75:      LSB %d/5 | DCT Pending\n", lsb_robust_jpeg);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}

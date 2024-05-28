#include "kernel_test.h"
#include "fp6_linear.cuh"

int main(int argc, char** argv)
{
    // Parsing the inputs from CLI.
    if (argc != 5) {
        printf("Wrong Inputs! Correct input format: ./kernel_test #Row_Weight #Column_Weight BatchSize SplitK\n");
        return -1;
    }
    size_t M_GLOBAL = atoi(argv[1]);
    size_t K_GLOBAL = atoi(argv[2]);
    size_t N_GLOBAL = atoi(argv[3]);
    int    SPLIT_K  = atoi(argv[4]);
    assert(M_GLOBAL%256==0);                 // Currently, M_GLOBAL must be a multiple of 256.
    assert(K_GLOBAL%64==0);                  // Currently, K_GLOBAL must be a multiple of 64.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Matrices in quantized FP6 models with faked values.
    unsigned char* A_6bit_h  = (unsigned char*)malloc(M_GLOBAL*K_GLOBAL*6/8);       CheckMallocCPU(A_6bit_h, __LINE__);     // Weight matrix with FP6 values, stored in row-major.
    for(size_t i=0; i<M_GLOBAL*K_GLOBAL*6/8; i++)   A_6bit_h[i] = rand() % 256;                                             // Random initialization.
    half*          A_Scale_h = (half*)malloc(M_GLOBAL*sizeof(half));                CheckMallocCPU(A_Scale_h, __LINE__);    // Quantization Scales with FP16 values.
    for(size_t i=0; i<M_GLOBAL; i++)                A_Scale_h[i] = float(rand()%256)/64.0f;                                 // Scale
    // Generaing FP16 format of the Weight Matrix
    half* A_16bit_h = (half*) malloc(M_GLOBAL*K_GLOBAL*sizeof(half));                           CheckMallocCPU(A_16bit_h, __LINE__);
    DeQuantMatrix_FP6_To_FP16(A_16bit_h, A_6bit_h, M_GLOBAL, K_GLOBAL, A_Scale_h);
    // In-place weight pre-packing
    weight_matrix_prepacking((int*)A_6bit_h, (int*)A_6bit_h, M_GLOBAL, K_GLOBAL);

    // Devices Memory
    unsigned char*  A_6bit;
    half*           A_Scale;
    half*           A_16bit;
    cudaMalloc(reinterpret_cast<void**>(&A_6bit),  M_GLOBAL*K_GLOBAL*6/8);             CheckMallocCUDA(A_6bit, __LINE__);
    cudaMalloc(reinterpret_cast<void**>(&A_Scale), M_GLOBAL*sizeof(half));             CheckMallocCUDA(A_Scale, __LINE__);
    cudaMalloc(reinterpret_cast<void**>(&A_16bit),          M_GLOBAL*K_GLOBAL*sizeof(half));    CheckMallocCUDA(A_16bit, __LINE__);
    // Memory Copy from CPU to GPU
    cudaMemcpy(A_6bit,     A_6bit_h,  M_GLOBAL*K_GLOBAL*6/8,          cudaMemcpyHostToDevice);
    cudaMemcpy(A_Scale,    A_Scale_h,          M_GLOBAL*sizeof(half),          cudaMemcpyHostToDevice);
    cudaMemcpy(A_16bit,             A_16bit_h,          M_GLOBAL*K_GLOBAL*sizeof(half), cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // B Matrix: Activations
    half* B_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL); CheckMallocCPU(B_h);       // col major 
    for (size_t i = 0; i < N_GLOBAL * K_GLOBAL; i++)
        B_h[i] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
    // Device memory
    half* B            = NULL;
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);               CheckMallocCUDA(B, __LINE__);
    // Memory Copy from CPU to GPU
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    checkLastCudaError(__LINE__);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    half* D_cublas = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_cublas), sizeof(half) * M_GLOBAL * N_GLOBAL);        CheckMallocCUDA(D_cublas, __LINE__);
    cudaMemset(D_cublas, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);
    //cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);          // Tensor core NOT enabled
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);             // Tensor core enabled
    cudaDeviceSynchronize();
    int              m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    const float      alpha     = 1.0;
    const float      beta      = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,   CUBLAS_OP_N,
                                     m, n, k,
                                     &alpha,
                                     A_16bit,   CUDA_R_16F, k,
                                     B,         CUDA_R_16F, k,
                                     &beta,
                                     D_cublas,  CUDA_R_16F, m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,   CUBLAS_OP_N,
                                     m, n, k,
                                     &alpha,
                                     A_16bit,   CUDA_R_16F, k,
                                     B,         CUDA_R_16F, k,
                                     &beta,
                                     D_cublas,  CUDA_R_16F, m,
                                     CUDA_R_32F,
                                     CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas = 0;
    cudaEventElapsedTime(&milliseconds_cublas, start, stop);
    milliseconds_cublas = milliseconds_cublas / BENCHMARK_ITERATION;
    float tflops_cublas = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cublas / 1000.)) / 1e12;
    //
    half* D_cublas_h = NULL;  // col major
    D_cublas_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);   CheckMallocCPU(D_cublas_h);
    cudaMemcpy(D_cublas_h, D_cublas, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_cublas);
    checkLastCudaError(__LINE__);
    /////////////////////////////////////////////////////////////////////////////////////////////////
    half* D_fp6 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_fp6), sizeof(half) * M_GLOBAL * N_GLOBAL); CheckMallocCUDA(D_fp6);
    cudaMemset(D_fp6, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    //
    int Split_K = SPLIT_K;
    float* Reduction_Workspace = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(float) * M_GLOBAL * N_GLOBAL * Split_K);   CheckMallocCUDA(Reduction_Workspace, __LINE__);
    //
    for (int i = 0; i < WARM_UP_ITERATION; i++)
        fp6_linear_kernel(  0,
                        (uint4*)A_6bit, A_Scale,
                        B,
                        D_fp6,
                        M_GLOBAL, N_GLOBAL, K_GLOBAL,
                        Reduction_Workspace,  
                        Split_K);
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        fp6_linear_kernel(  0,
                        (uint4*)A_6bit, A_Scale,
                        B,
                        D_fp6,
                        M_GLOBAL, N_GLOBAL, K_GLOBAL,
                        Reduction_Workspace,  
                        Split_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    //
    float milliseconds_fp6 = 0.0f;
    cudaEventElapsedTime(&milliseconds_fp6, start, stop);
    milliseconds_fp6 = milliseconds_fp6 / BENCHMARK_ITERATION;
    float tflops_fp6 = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_fp6 / 1000.)) / 1e12;
    half* D_fp6_h = NULL;  // col major
    D_fp6_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_fp6_h, D_fp6, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_fp6);
    cudaFree(Reduction_Workspace);
    /////////////////////////////////////////////////////////////////////////////////////////////////
    double totalRelativeError_fp6  = ComputeTotalError(D_cublas_h, D_fp6_h, M_GLOBAL, N_GLOBAL);
    printf("************************************* ");
    printf("[%d-bit Weights, e%dm%d] M: %d N: %d K: %d SplitK: %d", 6, 3, 2, M_GLOBAL, N_GLOBAL, K_GLOBAL, SPLIT_K);
    printf(" ************************************\n");
    PrintPerformance("cuBLAS", milliseconds_cublas, tflops_cublas, 0.0);
    PrintPerformance("fp6_llm", milliseconds_fp6, tflops_fp6, totalRelativeError_fp6);

    //PrintMismatch("fp6", 100, 0.002, D_cublas_h, D_fp6_h, M_GLOBAL, N_GLOBAL);

    free(D_cublas_h);
    free(D_fp6_h);
    cudaFree(B);
    return 0;
}

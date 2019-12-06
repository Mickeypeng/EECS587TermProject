#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <cstdlib>
#include <math.h>

#define BLOCK_SIZE 16
#define FEATURE_LEN 128
using namespace std;

void calMeanVar(double* v, double& mean, double& var){
    double sum = 0;
    for(int i=0;i<FEATURE_LEN;i++)
        sum+=v[i];
    mean =  sum / FEATURE_LEN;
    double sq_sum = 0;
    for(int i=0;i<FEATURE_LEN;i++)
        sq_sum+=v[i]*v[i];
    var = sqrt(sq_sum / FEATURE_LEN - mean * mean);
    //printf("%f,%f\n",mean,var);
    return;
}

void normalize(double* v){
    double mean = 0;
    double var = 0;
    calMeanVar(v, mean, var);
    for(int i = 0; i < FEATURE_LEN; i++){
        v[i] = (v[i] - mean) / var;
    }
    return;
}

cudaError_t MulWithCuda(double* A, double* B, double* C, int matrixSize);
__global__ void matrixMulCUDA(double *A, double *B, double *C, int blockRow, int blockCol,int featureNum)
{
    double ans=0;
    //Need to copy A[bx*BLOCK_SIZE:(bx+1)*BLOCK_SIZE][:]
    //Need to copy B[by*BLOCK_SIZE:(by+1)*BLOCK_SIZE][:]

    __shared__ double AC[BLOCK_SIZE][FEATURE_LEN];//BLOCK_SIZE*blockCol
    __shared__ double BC[BLOCK_SIZE][FEATURE_LEN];

    //Each Thread response for #blockCol copy

    for(int k=0;k<blockCol;k++)
    {
        if((blockIdx.x*BLOCK_SIZE+threadIdx.x)<featureNum&&(k*BLOCK_SIZE+threadIdx.y)<FEATURE_LEN)
            AC[threadIdx.x][k*BLOCK_SIZE+threadIdx.y]=A[(blockIdx.x*BLOCK_SIZE+threadIdx.x)*FEATURE_LEN+(k*BLOCK_SIZE+threadIdx.y)];
        //else
            //printf("blockIdx.x blockIdx.y threadIdx.x threadIdx.y %d %d %d %d %d %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,blockIdx.x*BLOCK_SIZE+threadIdx.x,k*BLOCK_SIZE+threadIdx.y);
        if((blockIdx.y*BLOCK_SIZE+threadIdx.x)<featureNum&&(k*BLOCK_SIZE+threadIdx.y)<FEATURE_LEN)
            BC[threadIdx.x][k*BLOCK_SIZE+threadIdx.y]=B[(blockIdx.y*BLOCK_SIZE+threadIdx.x)*FEATURE_LEN+(k*BLOCK_SIZE+threadIdx.y)];
        //else
            //printf("blockIdx.x blockIdx.y threadIdx.x threadIdx.y %d %d %d %d %d %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,blockIdx.x*BLOCK_SIZE+threadIdx.x,k*BLOCK_SIZE+threadIdx.y);
    }
    __syncthreads();

    /*
    for(int i=0;i<16;i++)
    {
        for(int j=0;j<128;j++)
            printf("%f ",BC[i][j]);
        printf("\n");
    }
*/
    //if(blockIdx.x==8&&blockIdx.y==8)
    //    printf("blockIdx.x blockIdx.y threadIdx.x threadIdx.y %d %d %d %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
    //if(blockIdx.x==8&&blockIdx.y==8&&threadIdx.x==0&&threadIdx.y==0)
    //        printf("%f",ans);
    for(int k=0;k<FEATURE_LEN;k++)
    {
        ans+=(AC[threadIdx.x][k]-BC[threadIdx.y][k])*(AC[threadIdx.x][k]-BC[threadIdx.y][k]);
    }

    //if(threadIdx.x==0&&threadIdx.y==0&&blockIdx.x==0&&blockIdx.y==0)
    //    printf("%f",ans);


    if((blockIdx.x*BLOCK_SIZE+threadIdx.x)<featureNum&&(blockIdx.y*BLOCK_SIZE+threadIdx.y)<featureNum)
        C[(blockIdx.x*BLOCK_SIZE+threadIdx.x)*featureNum+(blockIdx.y*BLOCK_SIZE+threadIdx.y)]=ans;
}

extern "C" int mulMatrix()
{
    int featureNum = 400;
    fstream fa("img1_400.txt",fstream::in);
    fstream fb("img2_400.txt",fstream::in);
    fstream fc("400_distance.txt",fstream::in);


    // if (argc == 1){
    //     printf("Please input your feature number!");
    //     return 1;
    // }else if (argc == 2){
    //     featureNum = atoi(argv[1]);
    //     printf("Input size: %d\n", featureNum);
    // }else{
    //     printf("Too many arguments!");
    //     return 1;
    // }

    double* A = (double*)malloc(featureNum * FEATURE_LEN * sizeof(double)); // We represent a 2D matrix in the form of 1D array.
    double* B = (double*)malloc(featureNum * FEATURE_LEN * sizeof(double)); // We represent a 2D matrix in the form of 1D array.
    double* C = (double*)malloc(featureNum * featureNum * sizeof(double)); // We represent a 2D matrix in the form of 1D array.

    // Initiate matrix A.
    for(int i = 0; i < featureNum; i++){
        for(int j = 0; j < FEATURE_LEN; j++){
            fa>>A[i * FEATURE_LEN + j];
            fb>>B[i * FEATURE_LEN + j];
        }
        normalize(A+i*FEATURE_LEN);
        normalize(B+i*FEATURE_LEN);
    }

    // Parallel.
    cudaError_t cudaStatus = MulWithCuda(A,B,C,featureNum);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "mulWithCuda failed!");return 1;}



    double err_max=0;
    printf("%f\n",C[0]);
    for(int i = 0; i < featureNum; i++){
        for(int j = 0; j < featureNum; j++){
            double tmp;
            fc>>tmp;
            err_max=err_max<abs(C[i*featureNum+j]-tmp*tmp)?C[i*featureNum+j]-tmp*tmp:err_max;

        }
    }
    printf("%f\n",err_max);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaDeviceReset failed!");return 1;}

    free(A);free(B);free(C);
    fa.close();fb.close();fc.close();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t MulWithCuda(double* A, double* B, double* C, int featureNum)
{
    double *dev_A;
    double *dev_B;
    double *dev_C;
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
    float gpu_time = 0.0f;

    // Launch a kernel on the GPU with one thread for each element.
    int blockRowNum=featureNum%BLOCK_SIZE?featureNum/BLOCK_SIZE+1:featureNum/BLOCK_SIZE;
    int blockColNum=FEATURE_LEN%BLOCK_SIZE?FEATURE_LEN/BLOCK_SIZE+1:FEATURE_LEN/BLOCK_SIZE;
    dim3 gridDime(blockRowNum,blockRowNum);
    dim3 blockDime(BLOCK_SIZE,BLOCK_SIZE);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two matrix
    cudaStatus = cudaMalloc((void**)&dev_A, featureNum * FEATURE_LEN * sizeof(double));
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc A failed!");goto Error;}

    cudaStatus = cudaMalloc((void**)&dev_B, featureNum * FEATURE_LEN * sizeof(double));
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc B failed!");goto Error;}

    cudaStatus = cudaMalloc((void**)&dev_C, featureNum * featureNum * sizeof(double));
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc C failed!");goto Error;}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_A, A, featureNum * FEATURE_LEN * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy A failed!");goto Error;}

    cudaStatus = cudaMemcpy(dev_B, B, featureNum * FEATURE_LEN * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy B failed!");goto Error;}

	// Set up timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	printf("Row,Col %d %d\n",blockRowNum,blockColNum);

    matrixMulCUDA<<<gridDime,blockDime>>>(dev_A,dev_B,dev_C,blockRowNum,blockColNum,featureNum);

    // copy result back
    cudaStatus = cudaMemcpy(C, dev_C, featureNum * featureNum * sizeof(double), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy C back failed!");goto Error;}



    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);goto Error;}

	// Close timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("Time spent: %.5f\n", gpu_time);
	cudaEventDestroy(start);
    cudaEventDestroy(stop);

Error:
    cudaFree(dev_A);cudaFree(dev_B);cudaFree(dev_C);
    return cudaStatus;
}

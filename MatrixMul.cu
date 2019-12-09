#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <cstdlib>
#include <math.h>

#define BLOCK_SIZE 16
#define FEATURE_LEN 128
using namespace std;
//kp[featureNum][4]
//h[3][3]->h[9]
__global__ void InlineCuda(double *kp, bool *choose, double* h,int featureNum,double thres) {
    unsigned int i = blockIdx.x*blockDim.x+ threadIdx.x;
    if(i<featureNum)
    {
        double p0=kp[featureNum*i];
        double p1=kp[featureNum*i+1];
        double ep0,ep1,ep2;
        ep0 = h[0]*p0 + h[1]*p1 + h[2];
        ep1 = h[3]*p0 + h[4]*p1 + h[5];
        ep2 = h[6]*p0 + h[7]*p1 + h[8];
        ep0=ep0/ep2;
        ep1=ep1/ep2;
        ep0=ep0-kp[featureNum*i+2];
        ep1=ep1-kp[featureNum*i+3];

        if(ep0*ep0+ep1*ep1<(thres*thres))
            choose[i]=true;
        else
            choose[i]=false;
    }
}
//fn2 power is the maxmimum power of 2 which fn2power<featureNum
__global__ void SumCuda(bool * g_idata,int* cnt, int featureNum,int fn2power)
{
    __shared__ unsigned int sdata[1024];
    unsigned int tid = threadIdx.x;//[0,1024)
    if(tid<fn2power)
    {
        sdata[tid]=g_idata[tid];
        if(tid+fn2power<featureNum)
            sdata[tid]+=g_idata[tid+fn2power];
        __syncthreads();
        // do reduction in shared mem
        for (unsigned int s=fn2power/2; s>0; s>>=1) {
            if (tid < s) {
                sdata[tid]+=sdata[tid+s];
            }
            __syncthreads();
        }
        // write result for this block to global mem
        if(tid==0)
        {
            cnt[0]=sdata[0];
        }

    }
}
//fn2 power is the maxmimum power of 2 which fn2power<featureNum
__global__ void FindMinCuda(double *g_idata, int *g_odata, int featureNum,int fn2power,double thres) {
    __shared__ double first[1024];
    __shared__ double second[1024];
    __shared__ int firstInd[1024];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;//[0,1024)
    unsigned int i = blockIdx.x*featureNum + threadIdx.x;
    double tmp;
    if(tid<fn2power)
    {
        first[tid]=g_idata[i];
        firstInd[tid]=tid;
        if(tid+fn2power<featureNum)
        {
            second[tid]=g_idata[i+fn2power];
            if(first[tid]>second[tid])
            {
                tmp=first[tid];
                first[tid]=second[tid];
                second[tid]=tmp;
                firstInd[tid]=tid+fn2power;
            }
        }
        else
        {
            second[tid]=20000;
        }

        __syncthreads();
        // do reduction in shared mem
        for (unsigned int s=fn2power/2; s>0; s>>=1) {
            if (tid < s) {
                if(first[tid]>first[tid+s])
                {

                    if(second[tid+s]>first[tid])
                        second[tid]=first[tid];
                    else
                        second[tid]=second[tid+s];

                    first[tid]=first[tid+s];
                    firstInd[tid]=firstInd[tid+s];
                }
                else
                {
                    if(second[tid]>first[tid+s])
                        second[tid]=first[tid+s];
                }
            }
            __syncthreads();
        }
        // write result for this block to global mem
        if(tid==0)
        {
            //printf("row:%d find:%d first: %f second: %f\n",blockIdx.x,firstInd[0],first[0],second[0]);
            if(first[0]/second[0]<(thres*thres))
                g_odata[blockIdx.x]=firstInd[0];
            else
                g_odata[blockIdx.x]=-1;
        }
        //if(tid==0) printf("%d sum:%f\n",blockIdx.x,tmp);
    }
}


__global__ void NormalizeCuda(double *g_idata) {
    __shared__ double sdata[FEATURE_LEN];
    __shared__ double asdata[FEATURE_LEN];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;//0-128
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    double tmp=g_idata[i];
    sdata[tid] = tmp;
    asdata[tid]=tmp*tmp;
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        else if(tid-64<s)
        {
            asdata[tid-64]+=asdata[tid-64+s];
        }
        __syncthreads();
    }
    // write result for this block to global mem

    double mean = sdata[0]/FEATURE_LEN;
    double var = sqrt(asdata[0]/ FEATURE_LEN - mean * mean);
    tmp=(tmp-mean)/var;
    g_idata[i]=tmp;
    //if(tid==0) printf("%d sum:%f\n",blockIdx.x,tmp);
}

cudaError_t MulWithCuda(double* A, double* B, double* C, int *IC, int matrixSize);
__global__ void CalDistanceCuda(double *A, double *B, double *C, int blockRow, int blockCol,int featureNum)
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

    for(int k=0;k<FEATURE_LEN;k++)
        ans+=(AC[threadIdx.x][k]-BC[threadIdx.y][k])*(AC[threadIdx.x][k]-BC[threadIdx.y][k]);

    if((blockIdx.x*BLOCK_SIZE+threadIdx.x)<featureNum&&(blockIdx.y*BLOCK_SIZE+threadIdx.y)<featureNum)
        C[(blockIdx.x*BLOCK_SIZE+threadIdx.x)*featureNum+(blockIdx.y*BLOCK_SIZE+threadIdx.y)]=ans;
}

int main(int argc, char *argv[])
{
    int featureNum;
    fstream fa("img1_400.txt",fstream::in);
    fstream fb("img2_400.txt",fstream::in);
    fstream fc("400_distance.txt",fstream::in);


    if (argc == 1){
        printf("Please input your feature number!");
        return 1;
    }else if (argc == 2){
        featureNum = atoi(argv[1]);
        printf("Input size: %d\n", featureNum);
    }else{
        printf("Too many arguments!");
        return 1;
    }

    double* A = (double*)malloc(featureNum * FEATURE_LEN * sizeof(double)); // We represent a 2D matrix in the form of 1D array.
    double* B = (double*)malloc(featureNum * FEATURE_LEN * sizeof(double)); // We represent a 2D matrix in the form of 1D array.
    double* C = (double*)malloc(featureNum * featureNum * sizeof(double)); // We represent a 2D matrix in the form of 1D array.
    int*   IC = (int*)   malloc(featureNum * sizeof(int));

    // Initiate matrix A.
    for(int i = 0; i < featureNum; i++){
        for(int j = 0; j < FEATURE_LEN; j++){
            fa>>A[i * FEATURE_LEN + j];
            fb>>B[i * FEATURE_LEN + j];
        }
        //normalize(A+i*FEATURE_LEN);
        //normalize(B+i*FEATURE_LEN);
    }

    // Parallel.
    cudaError_t cudaStatus = MulWithCuda(A,B,C,IC,featureNum);
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

// Helper function for using CUDA to find nearest
//A[featureNum, 128]
//B[featureNum, 128]
//C [featureNum,featureNum] is the distance of A,B
//indC[featureNum] is index, where -1 as unmatch
void MulWithCuda(double* A, double* B, int* indC, int featureNum, double thres)
{
    double *dev_A;
    double *dev_B;
    double *dev_C;
    int *mC;
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
    float gpu_time = 0.0f;

    // Launch a kernel on the GPU with one thread for each element.
    int blockRowNum=featureNum%BLOCK_SIZE?featureNum/BLOCK_SIZE+1:featureNum/BLOCK_SIZE;
    int blockColNum=FEATURE_LEN%BLOCK_SIZE?FEATURE_LEN/BLOCK_SIZE+1:FEATURE_LEN/BLOCK_SIZE;
    dim3 CalDistanceGridDim(blockRowNum,blockRowNum);
    dim3 CalDistanceBlockDim(BLOCK_SIZE,BLOCK_SIZE);
    dim3 NormalizeGridDim(featureNum);
    dim3 NormalizeBlockDim(FEATURE_LEN);
    dim3 FindMinGridDim(featureNum);
    dim3 FindMinBlockDim(1024);

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

    cudaStatus = cudaMalloc((void**)&mC, featureNum * sizeof(int));
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc mC failed!");goto Error;}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_A, A, featureNum * FEATURE_LEN * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy A failed!");goto Error;}

    cudaStatus = cudaMemcpy(dev_B, B, featureNum * FEATURE_LEN * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy B failed!");goto Error;}

	// Set up timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	NormalizeCuda<<<NormalizeGridDim,NormalizeBlockDim>>>(dev_A);
	NormalizeCuda<<<NormalizeGridDim,NormalizeBlockDim>>>(dev_B);
	// Is sync needed?
    CalDistanceCuda<<<CalDistanceGridDim,CalDistanceBlockDim>>>(dev_A,dev_B,dev_C,blockRowNum,blockColNum,featureNum);
    //TODO replace 512
    FindMinCuda<<<FindMinGridDim,FindMinBlockDim>>>(dev_C,mC,featureNum,512,thres);
    // Is sync needed?
    // copy result back
    cudaStatus = cudaMemcpy(indC, mC, featureNum * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy indC back failed!");goto Error;}

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);goto Error;}

	// Close timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	//printf("Time spent: %.5f\n", gpu_time);
	//for(int i=0;i<featureNum;i++)
        //printf("%d\n",indC[i]);

	cudaEventDestroy(start);
    cudaEventDestroy(stop);

Error:
    cudaFree(dev_A);cudaFree(dev_B);cudaFree(dev_C);cudaFree(mC);
}

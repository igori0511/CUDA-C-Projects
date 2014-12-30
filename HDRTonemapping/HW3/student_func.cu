

#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

// compute min max
__global__ void reduce_kernel(float * d_out,  float*  d_in, int op)
{
    //copy to shared memory
    extern  __shared__ float sdata[];
    
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId] ;
    __syncthreads();        // make sure entire block is loaded!
    
    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (tid < s) 
        {
            if(op){   
              sdata[tid] = min(sdata[tid], sdata[tid + s] );
            }    
            else{
              sdata[tid] = max(sdata[tid], sdata[tid + s] );  
            }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }
    if (tid == 0) 
    {
        d_out[blockIdx.x]  =  sdata[0];
    }
}

// compute histogram
__global__ void histo(unsigned int *d_bins,float *d_in,
                      float logLumMin,
                      float logLumRange,
                      const int numBins)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_in[myId] - logLumMin) / logLumRange * numBins));
    
    atomicAdd(&(d_bins[bin]), 1);
}

//compute cdf
__global__ void prescan(unsigned int *  h_odata, unsigned int *  h_indata, int n)
{
    extern  __shared__  unsigned int temp[]; // allocated on invocation 
    int thid = threadIdx.x; 
    int pout = 0, pin = 1; 
    // load input into shared memory.  
    // This is exclusive scan, so shift right by one and set first elt to 0 
    temp[pout*n + thid] = (thid > 0) ? h_indata[thid-1] : 0; 
    __syncthreads(); 
    for (int offset = 1; offset < n; offset *= 2) 
    { 
        pout = 1 - pout; // swap double buffer indices 
        pin  = 1 - pout; 
        if (thid >= offset) 
            temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset]; 
        else 
           temp[pout*n+thid] = temp[pin*n+thid]; 
        __syncthreads(); 
    } 
    h_odata[thid] = temp[pout*n+thid]; // write output 
}
    
// reduce 
void reduce (const float* const d_logLuminance,
             float *min_logLum,
             float *max_logLum,
             const size_t numRows,
             const size_t numCols)
{
  
    float *d_out;
    float *d_in;
    float *d_intermidiate;
    unsigned int numElem = numRows * numCols;
    size_t blocksize = numCols;
    size_t gridsize  = numRows;
    
    unsigned int size = numElem * sizeof(float);
    
    checkCudaErrors(cudaMalloc((void **)&d_in, size));
    checkCudaErrors(cudaMalloc((void **)&d_intermidiate,size));
    checkCudaErrors(cudaMalloc((void **)&d_out,sizeof(float)));
 
    
    // compute the min_max using reduce
    checkCudaErrors(cudaMemcpy(d_in,d_logLuminance,size,cudaMemcpyDeviceToDevice));
    
    reduce_kernel<<<gridsize,blocksize,blocksize * sizeof(float)>>>(d_intermidiate,d_in,1);
    
    reduce_kernel<<<1,blocksize,blocksize * sizeof(float)>>>(d_out,d_intermidiate,1);
 
    checkCudaErrors(cudaMemcpy(&(*min_logLum),d_out,sizeof(float),cudaMemcpyDeviceToHost));
    
    // compute the max using reduce
       
    reduce_kernel<<<gridsize,blocksize,blocksize * sizeof(float)>>>(d_intermidiate,d_in,0);
    
    reduce_kernel<<<1,blocksize,blocksize * sizeof(float)>>>(d_out,d_intermidiate,0);
 
    checkCudaErrors(cudaMemcpy(&(*max_logLum),d_out,sizeof(float),cudaMemcpyDeviceToHost));
    
    // free the memory
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_intermidiate)); 
    checkCudaErrors(cudaFree(d_out));
}    
//compute histo function
void histogram(const float* const d_logLuminance,
               unsigned int *h_histo,
               float min_logLum,
               float logLumRange,
               const size_t numRows,
               const size_t numCols,
               const size_t numBins)

{
     // step 3 generate a histogram
    // declare GPU memory pointers
    unsigned int * d_bins;
    float *d_in;
    unsigned int numElem = numRows * numCols;
    size_t blocksize  = numCols;
    size_t gridsize   = numRows;
    const size_t size_bins  =  numBins * sizeof(unsigned int);
    const size_t size_bytes = numElem * sizeof(float);
    
    
    // allocate GPU memory
    checkCudaErrors(cudaMalloc((void **) &d_bins, size_bins));
    checkCudaErrors(cudaMalloc((void **) &d_in,   size_bytes));
    // transfer arrays to the the GPU
    checkCudaErrors(cudaMemcpy(d_in,d_logLuminance,size_bytes,cudaMemcpyDeviceToDevice));
    //set memory to zeros
    checkCudaErrors(cudaMemset(d_bins,0x0,size_bins));
    //launch kernel
    histo<<<gridsize, blocksize>>>(d_bins,
                                   d_in,
                                   min_logLum,
                                   logLumRange,
                                   numBins);
    
    checkCudaErrors(cudaMemcpy(h_histo,d_bins,size_bins,cudaMemcpyDeviceToHost));
    
    cudaFree(d_in);
    cudaFree(d_bins);
    
    
}    
// main function
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    // reduce min_max
    reduce(d_logLuminance,&min_logLum,&max_logLum,numRows,numCols);
    // step 2 
    float logLumRange = max_logLum - min_logLum;
    // step 3 compute histogram
    const size_t size_bins  =  numBins * sizeof(unsigned int);
    unsigned int *h_histo = (unsigned int *)malloc(size_bins);
    
    histogram(d_logLuminance,h_histo,min_logLum,logLumRange,numRows,numCols,numBins);
    // step 4 
    // compute cdf
    unsigned int *d_in;
    
    checkCudaErrors(cudaMalloc((void **) &d_in, size_bins));
    
    checkCudaErrors(cudaMemcpy(d_in,h_histo,size_bins,cudaMemcpyHostToDevice));
    
    prescan<<<1,1024, 2 * size_bins >>>(d_cdf,
                                        d_in,
                                        numBins);

    
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
}
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <curand_kernel.h>
#include <time.h>

using namespace std;

__global__ void randInit(curandState* state){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(1337, idx, 0, &state[idx]);
}

__global__ void monteCarlo(float* forecasts, float mu, float sigma, int iterations, int periods, int dt){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float eps;

  for (int i = idx; i < iterations; i += stride){
    for (int j = 1; j < periods; j++){
      curandState s;
      curand_init(1337, idx + j, 0, &s);
      eps = curand_normal(&s);
      forecasts[(i * periods) + j] =  forecasts[(i * periods) + (j - 1)] * exp((mu - (pow(sigma, 2) * .5)) * dt + sigma * eps * sqrt((float) dt));
    }
  }

  return;
}

int main(void)
{
  clock_t start, end;
  start = clock();

  float s0;
  float mu;
  float sigma; 
  int startdate;
  int iterations; 
  int increments;
  int dt;
  int periods;

  string fname = "log_returns.csv";
  ifstream returns_file;
  returns_file.open(fname);

  if (returns_file.is_open()) { 
    string line;

    getline(returns_file, line);
    s0 = stof(line);

    getline(returns_file, line);
    mu = stof(line);

    getline(returns_file, line);
    sigma = stof(line);

    getline(returns_file, line);
    startdate = stof(line);

    getline(returns_file, line);
    iterations = stof(line);

    getline(returns_file, line);
    increments = stof(line);

    getline(returns_file, line);
    dt = stof(line);

    periods = increments / dt;

    getline(returns_file, line);
    int return_num = stof(line);
    float* log_returns = new float[return_num];
    
    for (int i = 0; i < return_num; i++)
    {
      getline(returns_file, line);
      log_returns[i] = stof(line);
    }
        
    returns_file.close(); 
  }
  else{
    cout << "Failed to open " << fname;
  }

  int nThreads = 256;
  int nBlocks = (iterations * periods + nThreads - 1) / nThreads;
  
  /*
  nThreads = 1;
  nBlocks = 1;
  */

  float* forecasts;
  cudaMallocManaged(&forecasts, iterations * periods * sizeof(float));
  for (int i = 0; i < iterations * periods; i += periods){
    forecasts[i] = s0;
  }

  //curandState* d_state;
  //cudaMalloc(&d_state, nThreads * nBlocks); 
  //randInit<<<nBlocks, nThreads>>>(d_state);
  monteCarlo<<<nBlocks, nThreads>>>(forecasts, mu, sigma, iterations, periods, dt); 

  cudaDeviceSynchronize();
  
  ofstream forecasts_file;
  forecasts_file.open("forecasts.csv");
  if (forecasts_file.is_open()){
    forecasts_file << startdate << "," << periods << "\n";
    for (int i = 0; i < iterations; i++){
      for (int j = 0; j < periods; j++){
        forecasts_file << forecasts[(i * periods) + j];
        if (j < periods - 1) {forecasts_file << ",";}
      }
      forecasts_file << endl;
    }
    forecasts_file.close();
  }
  else{
    cout << "Could not open forecast.csv";
  }

  cudaFree(forecasts);

  end = clock();
  cout << ((double) (end - start)) / CLOCKS_PER_SEC << "\n";
  
  cudaError_t err = cudaGetLastError();  // add
  if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; 
  return 0;
}
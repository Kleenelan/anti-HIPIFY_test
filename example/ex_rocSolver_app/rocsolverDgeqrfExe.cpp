
// example.cpp source code //

 
#include <algorithm> // for std::min
#include <stddef.h>  // for size_t
#include <stdio.h>
#include <vector>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
 
void init_vector(double* A, int n)
{
  for(int i=0; i<n; i++)
    A[i] = (rand()%2000)/1000.0;
}
 
void print_matrix(double* A, int M, int N, int lda)
{
  for(int i=0; i<M; i++)
  {
    for(int j=0; j<N; j++)
    {
      printf("%7.4f, ", A[i + j*lda]);
    }
    printf("\n");
  }
 
}
 
int main() {
  rocblas_int M = 7;
  rocblas_int N = 7;
  rocblas_int lda = M;
 
  // here is where you would initialize M, N and lda with desired values
 
  rocblas_handle handle;
  rocblas_create_handle(&handle);
 
  size_t size_A = size_t(lda) * N;          // the size of the array for the matrix
  size_t size_piv = size_t(std::min(M, N)); // the size of array for the Householder scalars
 
  std::vector<double> hA(size_A);      // creates array for matrix in CPU
  std::vector<double> hIpiv(size_piv); // creates array for householder scalars in CPU
 
  init_vector(hA.data(), size_A);
  memset(hIpiv.data(), 0, size_piv*sizeof(double));
 
  print_matrix(hA.data(), M, N, lda);
 
  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);      // allocates memory for matrix in GPU
  hipMalloc(&dIpiv, sizeof(double)*size_piv); // allocates memory for scalars in GPU
 
  // here is where you would initialize matrix A (array hA) with input data
  // note: matrices must be stored in column major format,
  //       i.e. entry (i,j) should be accessed by hA[i + j*lda]
 
  // copy data to GPU
  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);
  // compute the QR factorization on the GPU
  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);
  // copy the results back to CPU
  hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);
 
  printf("\nR =\n");
  print_matrix(hA.data(), M, N, lda);
  printf("\ntau=\n");
  print_matrix(hIpiv.data(), 1, N, 1);
 
  // the results are now in hA and hIpiv, so you can use them here
 
  hipFree(dA);                        // de-allocate GPU memory
  hipFree(dIpiv);
  rocblas_destroy_handle(handle);     // destroy handle
}


// Author: Patrick Wieschollek <mail@patwie.com>
#ifndef P2DIST_OP_HH
#define P2DIST_OP_HH

template<typename T>
void P2distOpForwardCudaKernelLauncher(T* top,
                                          const T* matrixA, const T* matrixB, 
                                          const int B, const int M, const int N, const int D);


#endif
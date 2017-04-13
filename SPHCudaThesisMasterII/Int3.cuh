#ifndef INT3_CUH
#define INT3_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

class Int3{
public:
	__host__ __device__ Int3();
	__host__ __device__ Int3(int _x, int _y, int _z);
	__host__ __device__ void Add(int _x, int _y, int _z);
	__host__ __device__ void Add(Int3 value);
	__host__ __device__ void Substract(int _x, int _y, int _z);
	__host__ __device__ void Substract(Int3 value);
	__host__ __device__ void Multiply(int val);
	__host__ __device__ void printValue();
	int x;
	int y;
	int z;
};

#endif


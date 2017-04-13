#ifndef FLOAT3_CUH
#define FLOAT3_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

class Float3{
public:
	__host__ __device__ Float3();
	__host__ __device__ Float3(float _x, float _y, float _z);
	__host__ __device__ void Add(int _x, int _y, int _z);
	__host__ __device__ void Add(Float3 value);
	__host__ __device__ Float3 AddFunc(Float3 value);
	__host__ __device__ void Add(float _x, float _y, float _z);
	__host__ __device__ void Substract(int _x, int _y, int _z);
	__host__ __device__ void Substract(float _x, float _y, float _z);
	__host__ __device__ void Substract(Float3 value);
	__host__ __device__ Float3 SubstractFunc(Float3 value);
	__host__ __device__ void Multiply(float val);
	__host__ __device__ Float3 MultiplyFunc(float val);
	__host__ __device__ void Multiply(int val);
	__host__ __device__ void Devide(float val);
	__host__ __device__ Float3 DevideFunc(float val);
	__host__ __device__ void Devide(int val);
	__host__ __device__ float Mag2();
	__host__ __device__ void printValue();
	float x;
	float y;
	float z;
};

#endif


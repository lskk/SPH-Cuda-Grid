/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

#include "float3.cuh"
*/
//==========================================================================================//
#include <stdio.h>
#include <stdlib.h>
#include <GL\glew.h>
//GLFW
#include <GLFW\glfw3.h>
#include <glm\vec3.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include "float3.cuh"
#include <iostream>
#include "Shader.cuh"
#include <windows.h>

//SSFRInclude
#include "SsfrParam.cuh"

//SPH
#include "SPHCompute.cuh"
#include "Scene.cuh"
#include "SPHParam.cuh"

//USING CAMERA
#include "Camera.cuh"

//Kernel
#include "KernelOption.cuh"

SsfrParam ssfr;
SPHParam sphparam;

void initParam(){
	ssfr.setParticleSize(400);
	sphparam.restDensity = 1.2f;
	sphparam.gasConstant = 200.f;
	sphparam.viscosityCoeff = 0.6f;// (){ return 6.5f; }
	sphparam.timeStep = (float)1.3f / 60;// (){ return 10 / 80; }
	sphparam.surfaceTreshold = 2.0f;// (){ return 6.0f; }
	sphparam.surfaceTensionCoeff = 1.0f;// (){ return 0.0728; }
	sphparam.gravity = vec3(0.0f, -9.8f, 0.0f);// (){ return vec3(0.0f, 9.8f, 0.0f); }
	sphparam.mParticleRadius = 0.1f;// (){ return 0.5; }
	sphparam.wallDamping = -0.2f;// (){ return 0.2; }
	sphparam.particleSizeReserver = 2200;// (){ return 100; }
	sphparam.particleCounter = 2240;
	sphparam.kernelSize = 1.5f;
	sphparam.maxParticleCount = 3000;
	sphparam.LOrandomizer = 0;
	sphparam.HIrandomizer = 0.1;
	gLight.ambient = vec4(0.0, 0.0, 0.0, 1.0);
	gLight.diffuse = vec4(1.0, 1.0, 1.0, 1.0);
	gLight.specular = vec4(1.0, 1.0, 1.0, 1.0);
	gLight.position = vec4(0.0, -5.0, 0.0, 1.0);
}

/*
int main(){
	initParam();
	//INISIASI SPH
	DamBreakScene dam(sphparam);
	dam.createScene();
	//dam.debugPosition();
    KernelOption kernel = KernelOption(sphparam.kernelSize);
	SPHCompute sph(sphparam, kernel, &dam.getAllParticles()[0]);
	while (true){
		sph.updateParticles();
	}
	getchar();
	return 0;
}
*/
/*
void initFloat3(Float3 *floatval, int size){
	for (int i = 0; i < size; i++){
		floatval[i].x = rand();
		floatval[i].y = rand();
		floatval[i].z = rand();
	}
}

void initFloat3(Float3 *floatval, float val, int size){
	for (int i = 0; i < size; i++){
		floatval[i].x = val;
		floatval[i].y = val;
		floatval[i].z = val;
	}
}

__global__ void addValue(Float3 *floatval, Float3 x, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size){
		floatval[i].Add(x);
	}
}

__global__ void printAll(Float3 *floatval, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size){
		floatval[i].printValue();
	}
}

__global__ void deviceOperation(Float3 *floatval, Float3 *floatval_n, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size){
	    for (int j = 0; j < size; j++){
		    floatval[i].Add(floatval_n[j]);
        }
	}
    __syncthreads();
}

int main(){
	int size = 64;
	dim3 block(2);
	dim3 thread(32);
	Float3 temp(1.0f, 1.0f, 1.0f);

	Float3 *floatval;
    Float3 *floatval_d;
    Float3 *floatval_d1;
	floatval = (Float3*)malloc(size*sizeof(Float3));
    cudaMalloc(&floatval_d, (size*sizeof(Float3)));
    cudaMalloc(&floatval_d1, (size*sizeof(Float3)));

	initFloat3(floatval, 1.0f, size);
    cudaMemcpy(floatval_d, floatval, size * sizeof(Float3), cudaMemcpyHostToDevice);
    cudaMemcpy(floatval_d1, floatval, size * sizeof(Float3), cudaMemcpyHostToDevice);

	//addValue<<<block,thread>>>(floatval_d, temp, size);
    deviceOperation<<<block,thread>>>(floatval_d, floatval_d1, size);
	//printAll<<<block,thread>>>(floatval, size);
    cudaMemcpy(floatval, floatval_d, size * sizeof(Float3), cudaMemcpyDeviceToHost);

    cudaFree(floatval_d);
    cudaFree(floatval_d1);
	getchar();
	return 0;
}
*/
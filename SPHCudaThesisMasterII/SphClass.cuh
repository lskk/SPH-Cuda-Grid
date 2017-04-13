#ifndef SPHCLASS_CUH
#define SPHCLASS_CUH

//SSFRInclude
#include "SsfrParam.cuh"

//SPH
#include "SPHCompute.cuh"
#include "Scene.cuh"
#include "SPHParam.cuh"

//Kernel
#include "KernelOption.cuh"

class SphClass{
public:
	SphClass();
	void initParam();
	void initSPH();
	void initCustomSPH(Particle* particle, int particlecount);
	void updateParticles();
	Particle *getAllLiquidParticles();
	void freeCuda();
	void freeCPU();
	SPHParam getSPHParam();
	SsfrParam getSsfrParam();
private:
	SsfrParam ssfr;
	SPHParam sphparam;
	Wall wall;
	SPHCompute sph;
};

#endif
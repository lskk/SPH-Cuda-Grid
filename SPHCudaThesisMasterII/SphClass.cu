#include "SphClass.cuh"

SphClass::SphClass(){}

void SphClass::initParam(){
	ssfr.setParticleSize(250);
	sphparam.surfaceColor = (vec4(1.00f, 0.95f, 0.00f, 1.0f));
	sphparam.insideColor = (vec4(0.94f, 0.22f, 0.15f, 1.0f));
	sphparam.restDensity = 2.2f;// 2.2 normal
	sphparam.gasConstant = 200.0f;
	sphparam.viscosityCoeff = 2.5f;// (){ return 6.5f; } 1.2 normal
	sphparam.timeStep = (float)1.3f / 60;// (){ return 10 / 80; }
	sphparam.surfaceTreshold = 0.1f;// (){ return 6.0f; }
	sphparam.surfaceTensionCoeff = 10.0f;// (){ return 0.0728; }
	sphparam.gravity = vec3(0.0f, -9.8f, 0.0f);// (){ return vec3(0.0f, 9.8f, 0.0f); }
	sphparam.mParticleRadius = 0.1f;// (){ return 0.5; }
	sphparam.wallDamping = 0.1f;// (){ return 0.2; }
	sphparam.particleCounter = 40960;//6144;//5120;//8192;//10240;
	sphparam.kernelSize = 1.5f;//1.5 default
	sphparam.maxParticleCount = 80000;
	sphparam.LOrandomizer = 0.0f;
	sphparam.HIrandomizer = 0.11f;
	sphparam.cudaThreadsPerblock = 128;
	wall.min = vec3(-10.0f, -2.0f, -2.0f);
	wall.max = vec3(30.0f, 20.0f, 20.0f);
}

void SphClass::initSPH(){
	//printf("lewat SINI");
	DamBreakScene dam(sphparam);
	dam.createScene();
	sphparam = dam.getParam();
    printf("%d particle initialized \n", sphparam.particleCounter);
	//dam.debugPosition();
	KernelOption kernel = KernelOption(sphparam.kernelSize);
	sph = SPHCompute(sphparam, kernel, &dam.getAllParticles()[0], wall);
	sph.initCudaMemoryAlloc();
	sph.initCudaGrid();
}

void SphClass::initCustomSPH(Particle *particles, int particlecount){
	if (particlecount < sphparam.maxParticleCount){
		sphparam.particleCounter = particlecount;
		KernelOption kernel = KernelOption(sphparam.kernelSize);
		sph = SPHCompute(sphparam, kernel, particles, wall);
		sph.initCudaMemoryAlloc();
        sph.initCudaGrid();
	}
	else{
		exit(0);
	}
}

void SphClass::updateParticles(){
	sph.updateParticles();
}

Particle* SphClass::getAllLiquidParticles(){
	return sph.getAllLiquidParticles();
}

void SphClass::freeCuda(){
	sph.freeParticles();
    sph.freeGrid();
}

void SphClass::freeCPU(){
	free(sph.getAllLiquidParticles());
}

SPHParam SphClass::getSPHParam(){
	return sphparam;
}

SsfrParam SphClass::getSsfrParam(){
	return ssfr;
}
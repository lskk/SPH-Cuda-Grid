#ifndef SCENE_CUH
#define SCENE_CUH
#include "Particle.cuh"
#include "SPHParam.cuh"
#include "Int3.cuh"
#include <stdio.h>
#include <iostream>
#include <malloc.h>

class DamBreakScene{
public:
	DamBreakScene(SPHParam par);
	void createScene();
	Particle* getAllParticles();
	Int3 getSize();
	float randomBetweenFloat(float LO, float HI);
	void debugPosition();
	SPHParam getParam();
private:
	Int3 size;
	Particle* particles;
	SPHParam param;
};

class CylinderScene{
public:
	CylinderScene(SPHParam param);
	Particle* getAllLiquidParticles();
private:
	Particle* particles;
	SPHParam sphparam;
};

#endif
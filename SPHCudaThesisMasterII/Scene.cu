#pragma once
#include "Scene.cuh"

DamBreakScene::DamBreakScene(SPHParam par){
    param = par;
	particles = (Particle *)malloc(param.maxParticleCount*sizeof(Particle));
}

Int3 DamBreakScene::getSize(){
	return size;
}

Particle* DamBreakScene::getAllParticles(){
	return particles;
}

void DamBreakScene::createScene(){
	//size = Int3(10, 10, 10);
	/*
	int startx = 1;
	int endx = 4;
	int starty = 6;
	int endy = 9;
	int startz = 1;
	int endz = 4;
	*/
	int startx = 0;
	int endx = 4;
	int starty = 0;
	int endy = 4;
	int startz = 0;
	int endz = 4;

	float multiply = 10;

	//float devide = 0.3f;
    float devide = 0.4f;
	int count = 0;
	for (int i = starty * multiply; i < endy*multiply; i++){
		for (int j = startz * multiply; j < endz*multiply; j++){
			for (int k = startx * multiply; k < endx*multiply; k++){
				if (count >= param.particleCounter){
					break;
				}
				Particle prt;
				prt.mMass = 1.f;
				prt.mVelocity = vec3(0.00, 0.00, 0.00);
				prt.mDensity = 0;
				prt.mPosition = vec3((float)k * devide, (float)i * devide, (float)j * devide);
				prt.color = vec4(0.94f, 0.22f, 0.15f, 1.0f);
				particles[count] = prt;
                //printf("%d. mPosition %f,%f,%f\n", count, particles[count].mPosition.x, particles[count].mPosition.y, particles[count].mPosition.z);
				count++;
			}
		}
	}
    param.particleCounter = count;
}

float DamBreakScene::randomBetweenFloat(float LO, float HI){
	return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}

SPHParam DamBreakScene::getParam(){
    return param;
}

void DamBreakScene::debugPosition(){
    for (int i = 0; i < param.particleCounter; i++){
		printf("mPosition %f,%f,%f\n", particles[i].mPosition.x, particles[i].mPosition.y, particles[i].mPosition.z);
	}
}


CylinderScene::CylinderScene(SPHParam param){
    sphparam = param;
}

Particle* CylinderScene::getAllLiquidParticles(){
    return particles;
}

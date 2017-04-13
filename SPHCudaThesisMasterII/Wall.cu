#include "Wall.cuh"

Wall::Wall(){
	min = vec3(-2.0f, -2.0f, -50.0f);
	max = vec3(8.0f, 20.0f, 10.0f);
}

Wall::Wall(vec3 _min, vec3 _max){
	min = _min;
	max = _max;
}
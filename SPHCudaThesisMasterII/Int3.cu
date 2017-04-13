#include "Int3.cuh"

Int3::Int3(){
}    

Int3::Int3(int _x, int _y, int _z){
	x = _x;
	y = _y;
	z = _z;
}

void Int3::Add(int _x, int _y, int _z){
	x = x + _x;
	y = y + _y;
	z = z + _z;
}

void Int3::Add(Int3 value){
	x = x + value.x;
	y = y + value.y;
	z = z + value.z;
}

void Int3::Substract(int _x, int _y, int _z){
	x = x - _x;
	y = y - _y;
	z = z - _z;
}

void Int3::Substract(Int3 value){
	x = x - value.x;
	y = y - value.y;
	z = z - value.z;
}

void Int3::Multiply(int val){
	x = x*val;
	y = y*val;
	z = z*val;
}

void Int3::printValue(){
	printf("(%d,%d,%d) \n", x, y, z);
}
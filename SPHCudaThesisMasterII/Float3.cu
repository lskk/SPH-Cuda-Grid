#include "float3.cuh"

Float3::Float3(){

}

Float3::Float3(float _x, float _y, float _z){
	x = _x;
	y = _y;
	z = _z;
}

void Float3::Add(Float3 value){
	x = x + value.x;
	y = y + value.y;
	z = z + value.z;
}

void Float3::Add(int _x, int _y, int _z){
	x = x + _x;
	y = y + _y;
	z = z + _z;
}

void Float3::Add(float _x, float _y, float _z){
	x = x + _x;
	y = y + _y;
	z = z + _z;
}

void Float3::Substract(int _x, int _y, int _z){
	x = x - _x;
	y = y - _y;
	z = z - _z;
}

void Float3::Substract(float _x, float _y, float _z){
	x = x - _x;
	y = y - _y;
	z = z - _z;
}

void Float3::Substract(Float3 value){
	x = x - value.x;
	y = y - value.y;
	z = z - value.z;
}

void Float3::Multiply(float val){
	x = x*val;
	y = y*val;
	z = z*val;
}

void Float3::Multiply(int val){
	x = x*val;
	y = y*val;
	z = z*val;
}

void Float3::Devide(float val){
	x = x / val;
	y = y / val;
	z = z / val;
}

void Float3::Devide(int val){
	x = x / val;
	y = y / val;
	z = z / val;
}

void Float3::printValue(){
	printf("(%f,%f,%f) \n", x, y, z);
}

Float3 Float3::MultiplyFunc(float val){
	return Float3(x*val, y*val, z*val);
}

Float3 Float3::DevideFunc(float val){
	return Float3(x/val, y/val, z/val);
}

Float3 Float3::AddFunc(Float3 value){
	return Float3(x + value.x, y + value.y, z + value.z);
}

Float3 Float3::SubstractFunc(Float3 value){
	return Float3(x - value.x, y - value.y, z - value.z);
}

float Float3::Mag2(){
	return (x*x + y*y + z*z);
}

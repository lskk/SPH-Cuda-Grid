#include "KernelOption.cuh"
#include "StandardConstant.cuh"

/*
KernelOption::KernelOption(){
}

KernelOption::KernelOption(float kernel){
	setKernelSize(kernel);
	setPoly6Value();
	setGradPoly6Value();
	setSpikyKernelValue();
	setLaplacianPoly6Value();
	setViscoKernelValue();
	kernel2 = kernelSize*kernelSize;
}

float KernelOption::getKernelSize() {
	return kernelSize;
}

void KernelOption::setKernelSize(float kernel){
	kernelSize = kernel;
}

void KernelOption::setPoly6Value(){
	poly6KernelValue = 315.0f / (64.0f * PI * pow(kernelSize, 9.0f));
}

void KernelOption::setGradPoly6Value(){
	gradPoly6KernelValue = -945.0f / (32.0f * PI * pow(kernelSize, 9.0f));
}

void KernelOption::setSpikyKernelValue(){
	//spikyKernelValue = 15.0f / (PI * pow(kernelSize, 6));
	spikyKernelValue = -45.0f / (PI * pow(kernelSize, 6));
}

void KernelOption::setViscoKernelValue(){
	//viscoKernelValue = 15.0f / (2* PI * pow(kernelSize, 3));
	viscoKernelValue = 45.0f / (PI * pow(kernelSize, 6.0f));
}

void KernelOption::setLaplacianPoly6Value(){
	laplacianPoly6Value = 945.0f / (32.0f * PI * pow(kernelSize, 9.0f));
}

float KernelOption::getPoly6Value(){
	return poly6KernelValue;
}

float KernelOption::getGradPoly6Value(){
	return gradPoly6KernelValue;
}

float KernelOption::getSpikyKernelValue(){
	return spikyKernelValue;
}

float KernelOption::getViscoKernelValue(){
	return viscoKernelValue;
}

float KernelOption::getLaplacianPoly6Value(){
	return laplacianPoly6Value;
}

vec3 KernelOption::getWPoly6Gradient(vec3 possDiff, float r2){
    float diff = kernel2 - r2;
	//return possDiff*(getGradPoly6Value()*powf(diff, 2));
	return possDiff*(getGradPoly6Value()*diff*diff);
	//return gradient;
}

float KernelOption::getWPoly6(float r2){
    float diff = kernel2 - r2;
	//return (getPoly6Value()*powf(diff,3));
	return (getPoly6Value()*diff*diff*diff);
}

float KernelOption::getWPoly6Laplacian(float r2){
	return getLaplacianPoly6Value()*(kernel2 - r2) * (7.0f*r2 - 3.0f*kernel2);
}

vec3 KernelOption::getWspikyGradient(vec3 possDiff, float r2){
	float diff = kernelSize - sqrt1(r2);
	//return possDiff*(getSpikyKernelValue()*powf(diff, 2));
	return possDiff*(getSpikyKernelValue()*diff*diff);
	//gradient = possDiff*(getSpikyKernelValue()*powf(kernelSize - r2, 2)) / (sqrt1(r2 + 0.001f));
	//gradient = possDiff.MultiplyFunc(getSpikyKernelValue()*powf(kernelSize - sqrt1(r2), 2));
	//return gradient;
}

float KernelOption::getWviscosityLaplacian(float r2){
	return getViscoKernelValue()*(kernelSize - sqrt1(r2));
	//return getViscoKernelValue()*(kernel2 - sqrt1(r2));
}

float KernelOption::getKernel2(){
    return kernel2;
}
*/


//WITH OUT PRECOMPUTATION

KernelOption::KernelOption(){
}

KernelOption::KernelOption(float kernel){
	setKernelSize(kernel);
	setPoly6Value();
	setGradPoly6Value();
	setSpikyKernelValue();
	setLaplacianPoly6Value();
	setViscoKernelValue();
	kernel2 = kernelSize*kernelSize;
}

float KernelOption::getKernelSize() {
	return kernelSize;
}

void KernelOption::setKernelSize(float kernel){
	kernelSize = kernel;
}

void KernelOption::setPoly6Value(){
	poly6KernelValue = 315.0f / (64.0f * PI * pow(kernelSize, 9.0f));
}

void KernelOption::setGradPoly6Value(){
	gradPoly6KernelValue = -945.0f / (32.0f * PI * pow(kernelSize, 9.0f));
}

void KernelOption::setSpikyKernelValue(){
	//spikyKernelValue = 15.0f / (PI * pow(kernelSize, 6));
	spikyKernelValue = -45.0f / (PI * pow(kernelSize, 6));
}

void KernelOption::setViscoKernelValue(){
	//viscoKernelValue = 15.0f / (2* PI * pow(kernelSize, 3));
	viscoKernelValue = 45.0f / (PI * pow(kernelSize, 6.0f));
}

void KernelOption::setLaplacianPoly6Value(){
	laplacianPoly6Value = 945.0f / (32.0f * PI * pow(kernelSize, 9.0f));
}

float KernelOption::getPoly6Value(){
	return 315.0f / (64.0f * PI * pow(kernelSize, 9.0f));
}

float KernelOption::getGradPoly6Value(){
	return -945.0f / (32.0f * PI * pow(kernelSize, 9.0f));
}

float KernelOption::getSpikyKernelValue(){
	return -45.0f / (PI * pow(kernelSize, 6));
}

float KernelOption::getViscoKernelValue(){
	return 45.0f / (PI * pow(kernelSize, 6.0f));
}

float KernelOption::getLaplacianPoly6Value(){
	return 945.0f / (32.0f * PI * pow(kernelSize, 9.0f));
}

vec3 KernelOption::getWPoly6Gradient(vec3 possDiff, float r2){
	float diff = kernel2 - r2;
	//return possDiff*(getGradPoly6Value()*powf(diff, 2));
	return possDiff*(getGradPoly6Value()*diff*diff);
	//return gradient;
}

float KernelOption::getWPoly6(float r2){
	float diff = kernel2 - r2;
	//return (getPoly6Value()*powf(diff,3));
	return (getPoly6Value()*diff*diff*diff);
}

float KernelOption::getWPoly6Laplacian(float r2){
	return getLaplacianPoly6Value()*(kernel2 - r2) * (7.0f*r2 - 3.0f*kernel2);
}

vec3 KernelOption::getWspikyGradient(vec3 possDiff, float r2){
	float diff = kernelSize - sqrt1(r2);
	//return possDiff*(getSpikyKernelValue()*powf(diff, 2));
	return possDiff*(getSpikyKernelValue()*diff*diff);
	//gradient = possDiff*(getSpikyKernelValue()*powf(kernelSize - r2, 2)) / (sqrt1(r2 + 0.001f));
	//gradient = possDiff.MultiplyFunc(getSpikyKernelValue()*powf(kernelSize - sqrt1(r2), 2));
	//return gradient;
}

float KernelOption::getWviscosityLaplacian(float r2){
	return getViscoKernelValue()*(kernelSize - sqrt1(r2));
	//return getViscoKernelValue()*(kernel2 - sqrt1(r2));
}

float KernelOption::getKernel2(){
	return kernel2;
}

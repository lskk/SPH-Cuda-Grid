#include "SPHCompute.cuh"

SPHCompute::SPHCompute(){}

SPHCompute::SPHCompute(SPHParam par, KernelOption kernel, Particle *liquidParticles, Wall wallin){
	param = par;
	kernelConfig = kernel;
	h_particles = liquidParticles;
    wall = wallin;
	setGridDimension();
}

void SPHCompute::setParam(SPHParam par){
    param = par;
}

void SPHCompute::setKernel(KernelOption kernel){
    kernelConfig = kernel;
}

void SPHCompute::setParticles(Particle *liquidParticles){
    h_particles = liquidParticles;
}

void SPHCompute::setGridDimension(){
	dimen_x = ceil((float) (wall.max.x - wall.min.x) / param.kernelSize);
	dimen_y = ceil((float) (wall.max.y - wall.min.y) / param.kernelSize);
	dimen_z = ceil((float) (wall.max.z - wall.min.z) / param.kernelSize);
    dimen_size = dimen_x * dimen_y * dimen_z;
	printf("Dimen x,y,z = %d,%d,%d \n",dimen_x,dimen_y,dimen_z);
}

void SPHCompute::setWall(Wall wallin){
    wall = wallin;
}

void SPHCompute::initCudaMemoryAlloc(){
    cudaMalloc(&d_particles, sizeof(Particle)*param.maxParticleCount);
}

void SPHCompute::initCudaHostToDev(){
    cudaMemcpy(d_particles, h_particles, param.particleCounter * sizeof(Particle), cudaMemcpyHostToDevice);
}

void SPHCompute::initCudaParticles(){
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	cudaStatus = cudaMalloc(&d_particles, sizeof(Particle)*param.maxParticleCount);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed? %s \n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(d_particles, h_particles, param.particleCounter * sizeof(Particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed? %s \n", cudaGetErrorString(cudaStatus));
	}
}

void SPHCompute::initCudaGrid(){
	cudaMalloc(&d_gridPos, sizeof(int)*param.maxParticleCount);
	h_gridPos = (int *)malloc(param.maxParticleCount*sizeof(int));
    cudaMalloc(&d_bin_count, sizeof(int)*dimen_size);
    h_bin_count = (int *)malloc(sizeof(int)*dimen_size);
	cudaMalloc(&d_prefSum, sizeof(int)*dimen_size);
	h_prefSum = (int *)malloc(sizeof(int)*dimen_size);
	cudaMalloc(&d_gridSorted, sizeof(int)*param.maxParticleCount);
	h_gridSorted = (int *)malloc(param.maxParticleCount*sizeof(int));
	cudaMalloc(&d_particleIdx, sizeof(int)*param.maxParticleCount);
	h_particleIdx = (int *)malloc(param.maxParticleCount*sizeof(int));
	cudaMalloc(&d_reference, sizeof(int)*dimen_size);
	h_reference = (int *)malloc(sizeof(int)*dimen_size);
}

void SPHCompute::addMoreParticles(Particle liquidParticle){
	int check_size = param.particleCounter + 1;
	if (check_size < param.maxParticleCount){
		h_particles[param.particleCounter] = liquidParticle;
	}
}

void SPHCompute::updateParticles(){
    initCudaHostToDev();
    dim3 threadsBin(param.cudaThreadsPerblock);
    dim3 blocksBin((dimen_size/param.cudaThreadsPerblock) + (dimen_size%param.cudaThreadsPerblock > 0? 1:0));
	dim3 threads(param.cudaThreadsPerblock);
	dim3 blocks((param.particleCounter / param.cudaThreadsPerblock) + (param.particleCounter%param.cudaThreadsPerblock > 0? 1:0)); 
	
	initialParticlesValuesKernel <<<blocks, threads >>>(d_particles, param);
	cudaDeviceSynchronize();

    // inisialisasi partikel ke dalam grid
	initGridPos << <blocks, threads >> >(d_particles, d_gridPos, param, dimen_x, dimen_y, dimen_z, wall); 
	cudaDeviceSynchronize();

    // inisialisasi nilai 0 untuk histogram
	binZeros<< <blocksBin, threadsBin >> >(d_bin_count, dimen_size); 
	cudaDeviceSynchronize();

    // menghitung histogram setiap partikel
    binCount<< <blocks, threads >> >(d_bin_count, d_gridPos, param); 
    cudaDeviceSynchronize();

    // exclusive scan untuk melakukan prefix sum
	device_ptr<int> d(d_bin_count);  
    device_vector<int> v(dimen_size);                    
    exclusive_scan(d, d + dimen_size, v.begin());
	d_prefSum = thrust::raw_pointer_cast(&v[0]);

    // sorting posisi grid dengan counting sort .. output : index partikel dan grid yang terurut
    csSortArray << <blocks, threads >> >(param, d_gridPos, d_prefSum, d_gridSorted, d_particleIdx);
	cudaDeviceSynchronize();

    // menentukan batas peralihan grid yg telah terurut dan menyimpannya di d_reference
	find_boundaries << <blocks, threads >> >(param.particleCounter, dimen_size, d_gridSorted, d_reference);
	cudaDeviceSynchronize();

	//copyDevToHost();
	//debugReference();
	//debugParticleIdxAndGridSorted();

	computeDensityAndPressureGridKernel << <blocks, threads >> >(d_particles, d_particleIdx, d_reference, param, kernelConfig, wall, dimen_x, dimen_y, dimen_z, dimen_size);
    cudaDeviceSynchronize();

	computeForcesGridKernel << <blocks, threads >> >(d_particles, d_particleIdx, d_reference, param, kernelConfig, wall, dimen_x, dimen_y, dimen_z, dimen_size);
	cudaDeviceSynchronize();

    computeNewPositionKernel <<<blocks, threads >>>(d_particles, param);
    cudaDeviceSynchronize();

	curandState* deviceStates = NULL;
	cudaMalloc(&deviceStates, param.particleCounter * sizeof(curandState));

	//initialise_curand <<<blocks, threads >>>(deviceStates, unsigned(time(NULL)));
	//cudaDeviceSynchronize();

	computeCollisionandBoundaryKernel <<<blocks, threads >>>(d_particles, param, wall);
	cudaDeviceSynchronize();

	//cudaFree(deviceStates);
    copyDevToHost();
}

Particle* SPHCompute::getAllLiquidParticles(){
	return h_particles;
}

Wall SPHCompute::getWall(){
	return wall;
}

void SPHCompute::copyDevToHost(){
	cudaMemcpy(h_particles, d_particles, param.particleCounter * sizeof(Particle), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_gridPos, d_gridPos, param.particleCounter * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_bin_count, d_bin_count, dimen_size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_prefSum, d_prefSum, dimen_size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_gridSorted, d_gridSorted, param.particleCounter * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particleIdx, d_particleIdx, param.particleCounter * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_reference, d_reference, dimen_size * sizeof(int), cudaMemcpyDeviceToHost);
}

void SPHCompute::debugGrid(){
	for (int i = 0; i < param.particleCounter; i++){
		printf("%d. mGrid %d [%f,%f,%f]\n",i , h_gridPos[i], h_particles[i].mPosition.x, h_particles[i].mPosition.y, h_particles[i].mPosition.z);
	}
}

void SPHCompute::debugBin(){
	for (int i = 0; i < dimen_size; i++){
		if (h_bin_count[i] > 0) printf("%d. bin %d \n",i , h_bin_count[i]);
	}
}

void SPHCompute::debugPrefSum(){
	for (int i = 0; i < dimen_size; i++){
		printf("%d. prefsum %d \n", i, h_prefSum[i]);
	}
}

void SPHCompute::debugReference(){
	for (int i = 0; i < dimen_size; i++){
		printf("%d. address_index_of particle ref %d \n", i, h_reference[i]);
	}
}

void SPHCompute::debugParticleIdxAndGridSorted(){
	for (int i = 0; i < param.particleCounter; i++){
		printf("%d. particle %d gridIdx %d \n", i, h_particleIdx[i], h_gridSorted[i]);
	}
}

void SPHCompute::debugPosition(){
	for (int i = 0; i < param.particleCounter; i++){
		printf("mPosition %f,%f%f\n", h_particles[i].mPosition.x, h_particles[i].mPosition.y, h_particles[i].mPosition.z);
	}
}

void SPHCompute::freeParticles(){
	cudaFree(d_particles);
}

void SPHCompute::freeGrid(){
    cudaFree(d_gridPos);
    free(h_gridPos);
    cudaFree(d_bin_count);
    free(h_bin_count);
	cudaFree(d_prefSum);
	free(h_prefSum);
	cudaFree(d_gridSorted);
	free(h_gridSorted);
	cudaFree(d_particleIdx);
	free(h_particleIdx);
	cudaFree(d_reference);
	free(h_reference);
}

__global__ void initGridPos(Particle *d_particles, int *d_gridPos, SPHParam param, int dimen_x, int dimen_y, int dimen_z, Wall wall){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < param.particleCounter){
		int pos_x, pos_y, pos_z;
		getGridFromParticlePos(d_particles[i], param.kernelSize, wall, dimen_x, dimen_y, dimen_z, &pos_x, &pos_y, &pos_z);
		d_gridPos[i] = getCell(dimen_x, dimen_y, dimen_z, pos_x, pos_y, pos_z);
	}
}

__global__ void binCount(int *d_bin_count, int *d_gridPos, SPHParam param){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < param.particleCounter){
		atomicAdd(&d_bin_count[d_gridPos[i]],1);
	}
}

__global__ void binZeros(int *d_bin_count, int bin_size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < bin_size){
		d_bin_count[i] = 0;
	}
}

__global__ void csSortArray(SPHParam param, int *d_gridPos, int* d_prefSum, int *d_gridSorted, int *d_particleIdx){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < param.particleCounter){
		int old_index = atomicAdd(&d_prefSum[d_gridPos[i]], 1);
		d_gridSorted[old_index] = d_gridPos[i];
		d_particleIdx[old_index] = i;
	}
}

__device__ int getCell(int dimen_x, int dimen_y, int dimen_z, int pos_x, int pos_y, int pos_z){
	return (pos_x * dimen_y * dimen_z) + (pos_y * dimen_z) + pos_z;
}

__device__ void getGridFromParticlePos(Particle p, float kernelSize, Wall wall, int dimen_x, int dimen_y, int dimen_z, int* pos_x, int* pos_y, int* pos_z){
	pos_x[0] = __float2int_rd((p.mPosition.x - wall.min.x) / kernelSize); //6
	pos_y[0] = __float2int_rd((p.mPosition.y - wall.min.y) / kernelSize); //1
	pos_z[0] = __float2int_rd((p.mPosition.z - wall.min.z) / kernelSize); //
	if (pos_x[0] < 0) { pos_x[0] = 0; }
	if (pos_y[0] < 0) { pos_y[0] = 0; }
	if (pos_z[0] < 0) { pos_z[0] = 0; }
	if (pos_x[0] >= wall.max.x) { pos_x[0] = wall.max.x-1; }
	if (pos_y[0] >= wall.max.y) { pos_y[0] = wall.max.y-1; }
	if (pos_z[0] >= wall.max.z) { pos_z[0] = wall.max.z-1; }
 
}

__global__ void initialParticlesValuesKernel(Particle *particles, SPHParam param){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < param.particleCounter){
		particles[i].mDensity = 0;
		particles[i].mPressure = 0;
		particles[i].mForces = vec3(0, 0, 0);
		particles[i].colorFieldNormal = vec3(0, 0, 0);
		particles[i].colorFieldLaplacian = 0.0f;
		particles[i].mMass = 1.f;
	}
	__syncthreads();
}

__global__ void computeDensityAndPressureGridKernel(Particle *particles, int *particleIdx, int *reference, SPHParam param, KernelOption kernel, Wall wall, int dimen_x, int dimen_y, int dimen_z, int dimen_size){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i >= param.particleCounter) return;

	int pos_x, pos_y, pos_z;
	getGridFromParticlePos(particles[i], param.kernelSize, wall, dimen_x, dimen_y, dimen_z, &pos_x, &pos_y, &pos_z);
	int gridIndex = getCell(dimen_x, dimen_y, dimen_z, pos_x, pos_y, pos_z);

	for (int a = -1; a <= 1; a++){
		for (int b = -1; b <= 1; b++){
			for (int c = -1; c <= 1; c++){
				int neigh_x = pos_x + a;
				int neigh_y = pos_y + b;
				int neigh_z = pos_z + c;

				if (isValidGrid(dimen_x, dimen_y, dimen_z, neigh_x, neigh_y, neigh_z)){
					int neighCellIndex = getCell(dimen_x, dimen_y, dimen_z, neigh_x, neigh_y, neigh_z);
					Query query_res = query_table(dimen_size, reference, neighCellIndex);
					neighbourGridDensity(particles, particleIdx, i, param, kernel, query_res);
				}
			}
		}
	}
	particles[i].mPressure = param.gasConstant*(particles[i].mDensity - param.restDensity);
}

__device__ void selfGridDensity(Particle *particles, int *partcleIdx, int i, SPHParam param, KernelOption kernel, Query queryresult){
    for (int j = queryresult.min ; j < queryresult.max; i++){
		vec3 r = particles[i].mPosition - (particles[partcleIdx[j]].mPosition);
		float r2 = r.x*r.x + r.y*r.y + r.z*r.z; //MAG2
		particles[i].mDensity = particles[i].mDensity + particles[partcleIdx[j]].mMass*kernel.getWPoly6(r2);
    }
}

__device__ void neighbourGridDensity(Particle *particles, int *partcleIdx, int i, SPHParam param, KernelOption kernel, Query query_res){
    while (query_res.min < query_res.max){
		//if (i == 323) printf("%d. neigh %d,%d,%d \n", i, particles[partcleIdx[query_res.min]].mPosition.x);
        vec3 r = particles[i].mPosition - (particles[partcleIdx[query_res.min]].mPosition);
        float r2 = r.x*r.x + r.y*r.y + r.z*r.z; //MAG2
		if (kernel.getKernel2() >= r2) particles[i].mDensity = particles[i].mDensity + particles[partcleIdx[query_res.min]].mMass*kernel.getWPoly6(r2);
        query_res.min++;
    }
}

__device__ bool isValidGrid(int dimen_x, int dimen_y, int dimen_z, int pos_x, int pos_y, int pos_z){
	if (pos_x >= dimen_x || pos_x < 0) return false;
	if (pos_y >= dimen_y || pos_y < 0) return false;
	if (pos_z >= dimen_z || pos_z < 0) return false;
	return true;
}

__global__ void computeForcesGridKernel(Particle *particles, int *particleIdx, int *reference, SPHParam param, KernelOption kernel, Wall wall, int dimen_x, int dimen_y, int dimen_z, int dimen_size){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= param.particleCounter) return;
	int pos_x, pos_y, pos_z;
	getGridFromParticlePos(particles[i], param.kernelSize, wall, dimen_x, dimen_y, dimen_z, &pos_x, &pos_y, &pos_z);
	int gridIndex = getCell(dimen_x, dimen_y, dimen_z, pos_x, pos_y, pos_z);
	for (int a = -1; a <= 1; a++){
		for (int b = -1; b <= 1; b++){
			for (int c = -1; c <= 1; c++){
				int neigh_x = pos_x + a;
				int neigh_y = pos_y + b;
				int neigh_z = pos_z + c;
				if (isValidGrid(dimen_x, dimen_y, dimen_z, neigh_x, neigh_y, neigh_z)){
					int neighCellIndex = getCell(dimen_x, dimen_y, dimen_z, neigh_x, neigh_y, neigh_z);
					Query query_res = query_table(dimen_size, reference, neighCellIndex);
					neighbourGridForces(particles, particleIdx, i, param, kernel, query_res);
				}
			}
		}
	}
	vec3 Fext = param.gravity*(particles[i].mDensity);
	particles[i].mForces += (Fext);
	//		particles[i].mNormal = (-1.0f)*colorFieldNormal;
	//Calculate surface tension 
	float colorFieldNormalMagnitude = sqrt1((particles[i].colorFieldNormal.x*particles[i].colorFieldNormal.x) + (particles[i].colorFieldNormal.y*particles[i].colorFieldNormal.y) + (particles[i].colorFieldNormal.z*particles[i].colorFieldNormal.z));
	if (colorFieldNormalMagnitude > param.surfaceTreshold){
		//float inv_norm = 1.0f / colorFieldNormalMagnitude;
		float inv_norm = reciprocal(colorFieldNormalMagnitude);
		float c = -param.surfaceTensionCoeff*particles[i].colorFieldLaplacian*inv_norm;

		//particles[i].mSurfaceFlag = true;
		vec3 Fsurfacetens = particles[i].colorFieldNormal*c;
		particles[i].mForces += (Fsurfacetens);
		particles[i].color = param.surfaceColor;
	}
	else{
		//particles[i].mSurfaceFlag = false;
		particles[i].color = param.insideColor;
	}
}

__device__ void selfGridForces(Particle *particles, int *partcleIdx, int i, SPHParam param, KernelOption kernel, Query queryresult){
	for (int j = queryresult.min; j < queryresult.max; i++){
		vec3 r = particles[i].mPosition - (particles[partcleIdx[j]].mPosition);
		float r2 = r.x*r.x + r.y*r.y + r.z*r.z; //MAG2
		vec3 spikyGradient = kernel.getWspikyGradient(r, r2);
		vec3 poly6Gradient = kernel.getWPoly6Gradient(r, r2);
		float V = particles[j].mMass / (particles[partcleIdx[j]].mDensity);
		float Vper2 = V*0.5f;
		vec3 Fpressure = spikyGradient*((Vper2)*(particles[i].mPressure + particles[partcleIdx[j]].mPressure));
		vec3 Fviscosity = (particles[partcleIdx[j]].mVelocity - (particles[i].mVelocity))*(V*kernel.getWviscosityLaplacian(r2));
		particles[i].mForces = particles[i].mForces + (-1.0f)*Fpressure + Fviscosity*param.viscosityCoeff;
		//LAPLACIAN NORMAL AND COLORFIELD
		particles[i].colorFieldNormal += poly6Gradient*V;
		particles[i].colorFieldLaplacian += (kernel.getWPoly6Laplacian(r2)*V);
	}
}

__device__ void neighbourGridForces(Particle *particles, int *partcleIdx, int i, SPHParam param, KernelOption kernel, Query query_res){
	while (query_res.min < query_res.max){
		//printf("particle %d - %d \n", i, partcleIdx[query_res.min]);
		vec3 r = particles[i].mPosition - (particles[partcleIdx[query_res.min]].mPosition);
		float r2 = r.x*r.x + r.y*r.y + r.z*r.z; //MAG2
		if (kernel.getKernel2() >= r2){
			vec3 spikyGradient = kernel.getWspikyGradient(r, r2);
			vec3 poly6Gradient = kernel.getWPoly6Gradient(r, r2);
			float V = particles[partcleIdx[query_res.min]].mMass / (particles[partcleIdx[query_res.min]].mDensity);
			float Vper2 = V*0.5f;
			vec3 Fpressure = spikyGradient*((Vper2)*(particles[i].mPressure + particles[partcleIdx[query_res.min]].mPressure));
			vec3 Fviscosity = (particles[partcleIdx[query_res.min]].mVelocity - (particles[i].mVelocity))*(V*kernel.getWviscosityLaplacian(r2));
			particles[i].mForces = particles[i].mForces + (-1.0f)*Fpressure + Fviscosity*param.viscosityCoeff;
			//LAPLACIAN NORMAL AND COLORFIELD
			particles[i].colorFieldNormal += poly6Gradient*V;
			particles[i].colorFieldLaplacian += (kernel.getWPoly6Laplacian(r2)*V);
		}
		query_res.min++;
	}
}

__global__ void computeNewPositionKernel(Particle *particles, SPHParam param){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < param.particleCounter){
        vec3 accell = particles[i].mForces/particles[i].mDensity;
        accell -= particles[i].mVelocity*param.wallDamping;
		particles[i].mVelocity += accell*param.timeStep;
		particles[i].mPosition += particles[i].mVelocity*param.timeStep;
	}
	__syncthreads();
}

__global__ void computeCollisionandBoundaryKernel(Particle *particles, SPHParam param, Wall wall){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < param.particleCounter){
        //float random = param.LOrandomizer +  (generate(globalState, i)) / ((1.0f / (param.HIrandomizer - param.LOrandomizer)));
		if (particles[i].mPosition.x + param.mParticleRadius >= wall.max.x)
		{
			particles[i].mVelocity.x = particles[i].mVelocity.x*(-(1.0f-param.wallDamping));
			particles[i].mPosition.x = wall.max.x;// - random;
		}

		if (particles[i].mPosition.x - param.mParticleRadius <= wall.min.x)
		{
			particles[i].mVelocity.x = particles[i].mVelocity.x*(-(1.0f-param.wallDamping));
			particles[i].mPosition.x = wall.min.x;// + random;
		}

		if (particles[i].mPosition.y + param.mParticleRadius >= wall.max.y)
		{
			particles[i].mVelocity.y = particles[i].mVelocity.y*(-(1.0f-param.wallDamping));
			particles[i].mPosition.y = wall.max.y;// - random;
		}

		if (particles[i].mPosition.y - param.mParticleRadius <= wall.min.y)
		{
			particles[i].mVelocity.y = particles[i].mVelocity.y*(-(1.0f-param.wallDamping));
			particles[i].mPosition.y = wall.min.y;// + random;
		}

		if (particles[i].mPosition.z + param.mParticleRadius >= wall.max.z)
		{
			particles[i].mVelocity.z = particles[i].mVelocity.z*(-(1.0f-param.wallDamping));
			particles[i].mPosition.z = wall.max.z;// - random;
		}

		if (particles[i].mPosition.z - param.mParticleRadius <= wall.min.z)
		{
			particles[i].mVelocity.z = particles[i].mVelocity.z*(-(1.0f-param.wallDamping));
			particles[i].mPosition.z = wall.min.z;// + random;
		}
	}
	__syncthreads();
}

__global__ void initialise_curand(curandState * state, unsigned long seed)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
	//printf("index idx = %d", idx);
	__syncthreads();
}

__device__ float generate(curandState* globalState, int ind)
{
	//copy state to local mem
	curandState localState = globalState[ind];
	//apply uniform distribution with calculated random
	float rndval = curand_uniform(&localState);
	//update state
	globalState[ind] = localState;
	//return value
	return rndval;
}

__global__ void find_boundaries(const int num_keys, const int num_bucket, const int *which_bucket, int *bucket_start){
	int index = threadIdx.x + blockIdx.x*blockDim.x +blockIdx.y*blockDim.x*gridDim.x;
	// Each thread looks at one entry in the sorted bucket index list
	if (index >= num_keys){
		return;
	}
	int previous_bucket = (index > 0 ? which_bucket[index - 1] : 0);
	int my_bucket = which_bucket[index];
	/*
	*/
	if (previous_bucket != my_bucket){
		for (int i = previous_bucket; i < my_bucket; ++i){
			bucket_start[i] = index;
		}
	}

	/*
	*/
	if (index == num_keys - 1){
		for (int i = my_bucket; i < num_bucket; ++i){
			bucket_start[i] = num_keys;
		}
	}
}

__device__ Query query_table(const int num_bucket, const int *bucket_start, const int key){
	const unsigned int bucket_id = key;
	const unsigned int list_start = (bucket_id > 0 ? bucket_start[bucket_id - 1] : 0);
	const unsigned int next_list_start = bucket_start[bucket_id];
	Query query(list_start, next_list_start);
	return query;
}

__global__ void queryDevice(const int num_bucket, const int *bucket_start, const int key){
	Query queryresult = query_table(num_bucket, bucket_start, key);
}

Particle *SPHCompute::getDeviceParticles(){
	return d_particles;
}

Particle *SPHCompute::getHostParticles(){
	return h_particles;
}

int *SPHCompute::getDeviceGridPositionUnsorted(){
	return d_gridPos;
}

int *SPHCompute::getHostGridPositionUnsorted(){
	return h_gridPos;
}

int *SPHCompute::getDeviceBin(){
	return d_bin_count;
}

int *SPHCompute::getHostBin(){
	return h_bin_count;
}

int *SPHCompute::getDeviceParticleIdSorted(){
	return d_particleIdx;
}

int *SPHCompute::getHostParticleIdSorted(){
	return h_particleIdx;
}

int *SPHCompute::getDeviceGridSorted(){
	return d_gridSorted;
}

int *SPHCompute::getHostGridSorted(){
	return d_gridSorted;
}

int SPHCompute::getDimenX(){
	return dimen_x;
}

int SPHCompute::getDimenY(){
	return dimen_y;
}

int SPHCompute::getDimenZ(){
	return dimen_z;
}

int SPHCompute::getDimenSize(){
	return dimen_size;
}
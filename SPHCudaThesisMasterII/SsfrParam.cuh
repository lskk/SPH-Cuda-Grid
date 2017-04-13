#ifndef SSFRPARAM_CUH
#define SSFRPARAM_CUH
#include <glm/vec4.hpp>

class SsfrParam
{
public:
	SsfrParam();
	~SsfrParam();
	float getParticleSize();
	void setParticleSize(float size);
private:
	float particleSize;
};

#endif
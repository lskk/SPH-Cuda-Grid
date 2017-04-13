#include "CustomMath.cuh"

__host__ __device__ float sqrt1(const float x)
{
	union
	{
		int i;
		float x;
	} u;
	u.x = x;
	u.i = (1 << 29) + (u.i >> 1) - (1 << 22);

	// Two Babylonian Steps (simplified from:)
	// u.x = 0.5f * (u.x + x/u.x);
	// u.x = 0.5f * (u.x + x/u.x);
	u.x = u.x + x / u.x;
	u.x = 0.25f*u.x + x / u.x;

	return u.x;
}

__host__ __device__ float reciprocal( float x ) {
    union {
        float dbl;
        unsigned uint;
    } u;
    u.dbl = x;
    u.uint = ( 0xbe6eb3beU - u.uint ) >> (unsigned char)1;
                                    // pow( x, -0.5 )
    u.dbl *= u.dbl;                 // pow( pow(x,-0.5), 2 ) = pow( x, -1 ) = 1.0 / x
    return u.dbl;
}

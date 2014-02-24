#pragma once

#include "helper_math.h"
#include "element.h"
#include <thrust/random.h>

__host__ __device__ __forceinline__ float3 texel(uchar4* tex, int2 size, float2 t) {
	int w = size.x, h = size.y;
	float x = t.x * w;
	float y = t.y * h;
	int ty = floorf(y);
	float fy = y - ty;

	ty %= h;
	int dy = ceilf(y);
	dy %= h;

	int lx = floorf(x);
	float fx = x - lx;

	lx %= w;
	int rx = ceilf(x);
	rx %= w;

	int idx0 = (ty*w+lx);
	int idx1 = (dy*w+lx);
	int idx2 = (ty*w+rx);
	int idx3 = (dy*w+rx);

	const float factor = 1.0/255.0;
	float3 p0 = make_float3(tex[idx0].x, tex[idx0].y, tex[idx0].z);
	float3 p1 = make_float3(tex[idx1].x, tex[idx1].y, tex[idx1].z);
	float3 p2 = make_float3(tex[idx2].x, tex[idx2].y, tex[idx2].z);
	float3 p3 = make_float3(tex[idx3].x, tex[idx3].y, tex[idx3].z);

	return (p0 * (1-fy) * (1-fx) + p1 * fy * (1-fx)
		+ p2 * (1-fy) * fx + p3 * fy * fx) * factor;
}

__host__ __device__ __forceinline__ float3 texel_supersample(uchar4* tex, int2 size, float2 t) {
	float3 c1 = texel(tex, size, t + make_float2(1e-8, 1e-8));
	float3 c2 = texel(tex, size, t + make_float2(   0, 1e-8));
	float3 c3 = texel(tex, size, t + make_float2(1e-8,    0));
	float3 c4 = texel(tex, size, t);
	return clamp((c1+c2+c3+c4)*0.25, 0.0f, 1.0f);
}

__host__ __device__ __forceinline__ unsigned int myhash(unsigned int a){
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

__host__ __device__ __forceinline__ float generateRandomNumberFromThread1(int2 resolution, float time, int x, int y){
	int index = x + (y * resolution.x);

	thrust::default_random_engine rng(myhash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return (float) u01(rng);
}

__host__ __device__ __forceinline__ float2 generateRandomNumberFromThread2(int2 resolution, float time, int x, int y){
	int index = resolution.y + x + (y * resolution.x);

	thrust::default_random_engine rng(myhash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return make_float2((float) u01(rng), (float) u01(rng));
}

__host__ __device__ __forceinline__ float2 generateRandomOffsetFromThread2(int2 resolution, float time, int x, int y){
	int index = resolution.y + x + (y * resolution.x);

	thrust::default_random_engine rng(myhash(index*time));
	thrust::uniform_real_distribution<float> u01(-0.5,0.5);

	return make_float2((float) u01(rng), (float) u01(rng));
}

__host__ __device__ __forceinline__ float3 generateRandomNumberFromThread(int2 resolution, float time, int x, int y){
	int index = x + (y * resolution.x);

	thrust::default_random_engine rng(myhash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return make_float3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

__host__ __device__ __forceinline__ float3 generateRandomOffsetFromThread(int2 resolution, float time, int x, int y){
	int index = x + (y * resolution.x);

	thrust::default_random_engine rng(myhash(index*time));
	thrust::uniform_real_distribution<float> u01(-0.5,0.5);

	return make_float3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

__host__ __device__ __forceinline__ float3 calculateRandomDirectionInSphere(float xi1, float xi2) {

	//crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
	const float TWO_PI = 2.0 * 3.14159265;
	float up = sqrt(xi1); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = xi2 * TWO_PI;

	return make_float3(up * sinf(around), over * sinf(around), cos(around));
}

__host__ __device__ __forceinline__ float3 calculateRandomDirectionInHemisphere_uniform(float3 normal, float xi1, float xi2) {
	float3 v = calculateRandomDirectionInSphere(xi1, xi2);
	if( dot(v, normal) < 0 ) return -v;
	else return v;
}

__host__ __device__ __forceinline__ float3 calculateRandomDirectionInHemisphere(float3 normal, float xi1, float xi2) {

	//return calculateRandomDirectionInHemisphere_uniform(normal, xi1, xi2);
	
	double r1=2*3.1415926535897932384626433832795*xi1, r2=xi2, r2s=sqrt(r2);
	float3 w = normalize(normal);
	float3 u = normalize(cross((fabs(w.x)>1e-6?make_float3(0, 1, 0):make_float3(w.x>0?-1:1, 0, 0)), w));
	float3 v = normalize(cross(w, u));
	return normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2));		

	/*
	//crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
	const float TWO_PI = 2.0 * 3.14159265;
	float up = sqrt(xi1); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = xi2 * TWO_PI;

	//Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
	const float SQRT_OF_ONE_THIRD = sqrtf(1.0/3.0);
	float3 directionNotNormal;
	
	if (fabsf(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = make_float3(1, 0, 0);
	} else if (fabsf(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = make_float3(0, 1, 0);
	} else {
		directionNotNormal = make_float3(0, 0, 1);
	}

	//Use not-normal direction to generate two perpendicular directions
	float3 perpendicularDirection1 = normalize(cross(normal, directionNotNormal));
	float3 perpendicularDirection2 = normalize(cross(normal, perpendicularDirection1));

	return ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 );
	*/
}

__host__ __device__ __forceinline__ float3 random3(int idx) {
	thrust::default_random_engine rng(idx);
	thrust::uniform_real_distribution<float> u01(0,1);	
	return make_float3(float(u01(rng))-0.5f, float(u01(rng))-0.5f, float(u01(rng))-0.5f);
}

__host__ __device__ __forceinline__ float3 pow(float3 v, float p) {
	return make_float3(powf(v.x, p), powf(v.y, p), powf(v.z, p));
}

__host__ __device__ __forceinline__ float3 mul(float m[9], float3 v) {
	return make_float3(
		m[0] * v.x + m[1] * v.y + m[2] * v.z,
		m[3] * v.x + m[4] * v.y + m[5] * v.z,
		m[6] * v.x + m[7] * v.y + m[8] * v.z
		);
}

__host__ __device__ __forceinline__ float3 refract(float3 i, float3 n, float eta) {
	float nDi = dot(n, i);
	float k = 1.0 - eta * eta * (1.0 - nDi * nDi);
	if (k < 0.0)
		return make_float3(0.0f);
	else
		return normalize(eta * i - (eta * nDi + sqrtf(k)) * n);
}

__host__ __device__ __forceinline__ float3 fminf(float3 v, float f) {
	return make_float3(fminf(v.x, f), fminf(v.y, f), fminf(v.z, f));
}

__host__ __device__ __forceinline__ float3 fmaxf(float3 v, float f) {
	return make_float3(fmaxf(v.x, f), fmaxf(v.y, f), fmaxf(v.z, f));
}

__host__ __device__ __forceinline__ float3 mix(float3 u, float3 v, float f) {
	return u * (1.0f-f) + v * f;
}

__host__ __device__ __forceinline__ float3 mix(float3 u, float3 v, float3 w, float alpha, float beta, float gamma) {
	return alpha * u + beta * v + gamma * w;
}

__host__ __device__ __forceinline__ float step(float edge, float v) {
	return (v > edge)?1.0f:0.0;
}

__host__ __device__ __forceinline__ float3 step(float3 edge, float3 u) {
	return make_float3(u.x>edge.x?1.0f:0.0, u.y>edge.y?1.0f:0.0, u.z>edge.z?1.0f:0.0);
}

__host__ __device__ __forceinline__ float filter(float v, float lower, float upper) {
	if(v>=lower && v<=upper) return 1.0;
	else return 0.0;
}

__host__ __device__ __forceinline__ float toonify(float v, int steps) {
	float s = 1.0 / steps;
	return floor(v / s) * s;
}

__host__ __device__ __forceinline__ float intensity(float3 c) {
	return 0.2989 * c.x + 0.5870 * c.y + 0.1140 * c.z;
}

__host__ __device__ __forceinline__ float3 toonify(float3 v, int steps) {
	float I = intensity(v);
	float Iout = toonify(I, steps);
	return v * Iout / I;
}

__host__ __device__ __forceinline__ mat3 outerProduct(const vec3& u, const vec3& v) {
	return mat3(
		u.x * v.x, u.x * v.y, u.x * v.z,
		u.y * v.x, u.y * v.y, u.y * v.z,
		u.z * v.x, u.z * v.y, u.z * v.z
		);
}

__device__ __forceinline__ float2 spheremap(float3 p) {
	const float PI = 3.1415926536;
	return make_float2((atanf(p.z / p.x) / PI + 1.0) * 0.5,
                -((asinf(p.y) / PI + 0.5)));
}


__device__ __forceinline__ float3 sphere_tangent(float3 p) {
	float phi = asinf(p.y);
	
	float2 bn = normalize(make_float2(p.x, p.z)) * sinf(phi);
	return make_float3(bn.x, -cosf(phi), bn.y);
}
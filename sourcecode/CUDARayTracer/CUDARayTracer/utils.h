#pragma once

#include "helper_math.h"
#include "element.h"
#include <thrust/random.h>

__host__ __device__ __inline__ float3 texel(uchar4* tex, int2 size, float2 t) {
	int w = size.x, h = size.y;
	float x = t.x * w;
	float y = t.y * h;
	int ty = floorf(y);
	ty %= h;
	int dy = ceilf(y);
	dy %= h;

	int lx = floorf(x);
	lx %= w;
	int rx = ceilf(x);
	rx %= w;

	float fx = x - lx;
	float fy = y - ty;

	float r = tex[ty*w+lx].x * ((1-fy) * (1-fx))
		+ tex[dy*w+lx].x * (fy * (1-fx))
		+ tex[ty*w+rx].x * ((1-fy) * fx)
		+ tex[dy*w+rx].x * (fy * fx);

	float g = tex[ty*w+lx].y * ((1-fy) * (1-fx))
		+ tex[dy*w+lx].y * (fy * (1-fx))
		+ tex[ty*w+rx].y * ((1-fy) * fx)
		+ tex[dy*w+rx].y * (fy * fx);

	float b = tex[ty*w+lx].z * ((1-fy) * (1-fx))
		+ tex[dy*w+lx].z * (fy * (1-fx))
		+ tex[ty*w+rx].z * ((1-fy) * fx)
		+ tex[dy*w+rx].z * (fy * fx);

	return clamp(make_float3(r, g, b), 0.0, 1.0);
}

__host__ __device__ __inline__ float3 texel_supersample(uchar4* tex, int2 size, float2 t) {
	float3 c1 = texel(tex, size, t + make_float2(1e-8, 1e-8));
	float3 c2 = texel(tex, size, t + make_float2(   0, 1e-8));
	float3 c3 = texel(tex, size, t + make_float2(1e-8,    0));
	float3 c4 = texel(tex, size, t);
	return (c1+c2+c3+c4)*0.25;
}

__host__ __device__ __inline__ float3 random3(int idx) {
	thrust::default_random_engine rng(idx);
	thrust::uniform_real_distribution<float> u01(0,1);
	return make_float3(float(u01(rng))-0.5f, float(u01(rng))-0.5f, float(u01(rng))-0.5f);
}

__host__ __device__ __inline__ float3 mul(float m[9], float3 v) {
	return make_float3(
		m[0] * v.x + m[1] * v.y + m[2] * v.z,
		m[3] * v.x + m[4] * v.y + m[5] * v.z,
		m[6] * v.x + m[7] * v.y + m[8] * v.z
		);
}

__host__ __device__ __inline__ float3 fminf(float3 v, float f) {
	return make_float3(fminf(v.x, f), fminf(v.y, f), fminf(v.z, f));
}

__host__ __device__ __inline__ float3 fmaxf(float3 v, float f) {
	return make_float3(fmaxf(v.x, f), fmaxf(v.y, f), fmaxf(v.z, f));
}

__host__ __device__ __inline__ float3 mix(float3 u, float3 v, float f) {
	return u * (1.0f-f) + v * f;
}

__host__ __device__ __inline__ float3 mix(float3 u, float3 v, float3 w, float alpha, float beta, float gamma) {
	return alpha * u + beta * v + gamma * w;
}

__host__ __device__ __inline__ float step(float edge, float v) {
	return (v > edge)?1.0f:0.0;
}

__host__ __device__ __inline__ float3 step(float3 edge, float3 u) {
	return make_float3(u.x>edge.x?1.0f:0.0, u.y>edge.y?1.0f:0.0, u.z>edge.z?1.0f:0.0);
}

__host__ __device__ __inline__ float filter(float v, float lower, float upper) {
	if(v>=lower && v<=upper) return 1.0;
	else return 0.0;
}

__host__ __device__ __inline__ float toonify(float v, int steps) {
	float s = 1.0 / steps;
	return floor(v / s) * s;
}

__host__ __device__ __inline__ float intensity(float3 c) {
	return 0.2989 * c.x + 0.5870 * c.y + 0.1140 * c.z;
}

__host__ __device__ __inline__ float3 toonify(float3 v, int steps) {
	float I = intensity(v);
	float Iout = toonify(I, steps);
	return v * Iout / I;
}

__host__ __device__ __inline__ mat3 outerProduct(const vec3& u, const vec3& v) {
	return mat3(
		u.x * v.x, u.x * v.y, u.x * v.z,
		u.y * v.x, u.y * v.y, u.y * v.z,
		u.z * v.x, u.z * v.y, u.z * v.z
		);
}

__device__ __inline__ float2 spheremap(float3 p) {
	const float PI = 3.1415926536;
	return make_float2((atanf(p.z / p.x) / PI + 1.0) * 0.5,
                -((asinf(p.y) / PI + 0.5)));
}


__device__ __inline__ float3 sphere_tangent(float3 p) {
	float phi = asinf(p.y);
	
	float2 bn = normalize(make_float2(p.x, p.z)) * sinf(phi);
	return make_float3(bn.x, -cosf(phi), bn.y);
}
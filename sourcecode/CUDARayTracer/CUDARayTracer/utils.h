#pragma once

#include "helper_math.h"

#include "element.h"

__host__ __device__ __inline__ float dot(const vec3& u, const vec3& v) {
	return u.dot(v);
}

__host__ __device__ __inline__ vec3 cross(const vec3& u, const vec3& v) {
	return u.cross(v);
}

__host__ __device__ __inline__ vec3 reflect(const vec3& u, const vec3& v) {
	return u - 2.0f * dot(u, v) * v;
}

__host__ __device__ __inline__ vec2 clamp(const vec2& v, float lower, float upper) {
	return vec2(
		clamp(v.x, lower, upper), 
		clamp(v.y, lower, upper)
		);
}

__host__ __device__ __inline__ vec3 clamp(const vec3& v, float lower, float upper) {
	return vec3(
		clamp(v.x, lower, upper), 
		clamp(v.y, lower, upper), 
		clamp(v.z, lower, upper)
		);
}

__host__ __device__ __inline__ vec3 normalize(const vec3& v) {
	return v.normalized();
}

__host__ __device__ __inline__ vec3 min(const vec3& v, float f) {
	return vec3(min(v.x, f), min(v.y, f), min(v.z, f));
}

__host__ __device__ __inline__ vec3 max(const vec3& v, float f) {
	return vec3(max(v.x, f), max(v.y, f), max(v.z, f));
}

__host__ __device__ __inline__ vec3 mix(const vec3& u, const vec3& v, float f) {
	return u * (1.0f-f) + v * f;
}

__host__ __device__ __inline__ vec3 step(const vec3& edge, const vec3& u) {
	return vec3(u.x>edge.x?1.0f:0.0, u.y>edge.y?1.0f:0.0, u.z>edge.z?1.0f:0.0);
}

__host__ __device__ __inline__ mat3 outerProduct(const vec3& u, const vec3& v) {
	return mat3(
		u.x * v.x, u.x * v.y, u.x * v.z,
		u.y * v.x, u.y * v.y, u.y * v.z,
		u.z * v.x, u.z * v.y, u.z * v.z
		);
}

__device__ __inline__ vec2 spheremap(vec3 p) {
	const float PI = 3.1415926536;
	return vec2((atanf(p.z / p.x) / PI + 1.0) * 0.5,
                -((asinf(p.y) / PI + 0.5)));
}


__device__ __inline__ vec3 sphere_tangent(vec3 p) {
	float phi = asinf(p.y);
	
	vec2 bn = vec2(p.x, p.z).normalized() * sinf(phi);
	return vec3(bn.x, -cosf(phi), bn.y);
}
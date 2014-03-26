#pragma once

#include "utils.h"
#include "perlin.cuh"

__device__ PerlinNoise pn;

__device__ int juliaset_iteration(float u, float v) {
	float2 z = make_float2(u, v);
	float2 cval = make_float2(-0.8, 0.16);

	float val = complex_mag(z);
	int counter = 0;
	const int maxiters = 8192;
	const float THRES = 65536;
	for(int i=0; i<maxiters; i++)
	{
		z = complex_mul(z, z) + cval;
		val = complex_mag(z);
		if( val >= THRES || i > maxiters ) break;
		counter++;
	}

	return counter;
}

__device__ float3 juliaset(float u, float v) {
	int counter = juliaset_iteration(u, v);

	const int maxiters = 8192;
	float t = counter / float(maxiters);
	t = powf(t, 0.25);
	float3 color;
	float3 c1 = make_float3(0);
	float3 c2 = make_float3(0.05, 0.10, 0.05);
	float3 c3 = make_float3(0.075, 0.15, 0.175);
	float3 c4 = make_float3(0.65, 0.75, 1.0);

	if( t >= 0.75 )
		color = mix(c2, c1, pow((t - 0.75) / 0.25, 1.0));
	else if( t >= 0.5 )
		color = mix(c3, c2, pow((t - 0.5) / 0.25, 0.25));
	else
		color = mix(c4, c3, pow(t / 0.5, 2.0));

	return color;
}

__device__ int juliaset_iteration_3d(float u, float v, float w) {
	float3 z = make_float3(u, v, w);
	float3 cval = make_float3(-0.8, 0.16, 0.1);

	float val = complex_mag(z);
	int counter = 0;
	const int maxiters = 8192;
	const float THRES = 65536;
	for(int i=0; i<maxiters; i++)
	{
		z = complex_mul(z, z) + cval;
		val = complex_mag(z);
		if( val >= THRES || i > maxiters ) break;
		counter++;
	}

	return counter;
}

__device__ float3 juliaset3d(float u, float v, float w) {
	int counter = juliaset_iteration_3d(u, v, w);

	const int maxiters = 8192;
	float t = counter / float(maxiters);
	t = powf(t, 0.25);
	float3 color;
	float3 c1 = make_float3(0);
	float3 c2 = make_float3(0.05, 0.10, 0.05);
	float3 c3 = make_float3(0.075, 0.15, 0.175);
	float3 c4 = make_float3(0.65, 0.75, 1.0);

	if( t >= 0.75 )
		color = mix(c2, c1, pow((t - 0.75) / 0.25, 1.0));
	else if( t >= 0.5 )
		color = mix(c3, c2, pow((t - 0.5) / 0.25, 0.25));
	else
		color = mix(c4, c3, pow(t / 0.5, 2.0));

	return color;
}

__device__ float3 perlin(float3 p) {
	p = p * 16.0;
	float vec[3] = {p.x, p.y, p.z};
	float x = (pn.noise3(vec)+1.0)*0.5;
	return make_float3(x, x, x);
}

__device__ float3 perlin2d(float u, float v) {
	const float scale = 64.0;
	float vec[2] = {u * scale * 2.0, v * scale};
	float x = (pn.noise2(vec)+1.0)*0.5;
	return make_float3(x, x, x);
}

__device__ float turbulence(float3 p, float size) {
    float value = 0.0, initialSize = size;
    
    while(size >= 1)
    {
		float vec[3] = {p.x/size, p.y/size, p.z/size};
		value += pn.noise3(vec) * size;
        size /= 2.0;
    }
    
    return(0.5 * value / initialSize);
}

__device__ float3 marble(float3 p) {
	const float scale = 32.0;

	float x = (cosf(p.x * 5.0 + turbulence(p*scale*4.0, scale) * 15.0)+1.0)*0.5 + 0.25;
	return clamp(make_float3(x*1.5, x*1.25, x*0.75), 0, 1.0);
}

__device__ float3 woodgrain(float3 p) {
	const float scale = 32.0;
	float x = fabsf(cosf( sqrtf(p.x*p.x + p.y*p.y) * 15.0 + turbulence(p*scale*1.5, scale) * 5.0) );

	const float3 light = make_float3(0.72, 0.72, 0.45);
	const float3 dark = make_float3(0.49, 0.33, 0.11);
	return mix(light, dark, x);
}

__device__ float3 chessboard(float2 t) {
	const int blocks = 8;
	int blkx = t.x*blocks;
	int blky = t.y*blocks;

	int flag = (blkx&0x1) ^ (blky&0x1);
	const float3 red = make_float3(1, 0, 0);
	const float3 yellow = make_float3(1, 1, 0);
	return mix(red, yellow, flag);
}

__device__ float3 chessboard3d(float3 p) {
	const int blocks = 8;
	int blkx = p.x*blocks;
	int blky = p.y*blocks;
	int blkz = p.z*blocks;

	int flag1 = (blkx&0x1) ^ (blky&0x1);
	int flag2 = (blkz&0x1) ^ (blky&0x1);
	const float3 red = make_float3(1, 0, 0);
	const float3 green = make_float3(0, 1, 0);
	const float3 blue = make_float3(0, 0, 1);
	return mix(red, green, blue, (flag1&flag2)?0.0:((flag1^flag2)?0.5:1.0));
}

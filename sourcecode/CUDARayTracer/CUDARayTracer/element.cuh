#pragma once

class vec3 {
public:
	__device__ __host__ vec3():x(0), y(0), z(0){}
	__device__ __host__ vec3(float x, float y, float z):x(x), y(y), z(z){}
	__device__ __host__ vec3(const vec3& v):x(v.x),y(v.y), z(v.z){}
	__device__ __host__ vec3& operator=(const vec3& v){
		x = v.x; y = v.y; z = v.z;
		return (*this);
	}

	__device__ __host__ vec3 cross(const vec3& v) {
		return vec3(
			y*v.z - z * v.y,
			z*v.x - x * v.z,
			x*v.y - y * v.x
			);
	}

	__device__ __host__ float dot(const vec3& v) {
		return x * v.x + y * v.y + z * v.z;
	}

	union {
		float3 data;
		struct {float x, y, z;};
		struct {float r, g, b;};
	};
};

class vec4 {
	float dot(const vec4& v) {
	}

	union {
		float4 data;
		struct {float x, y, z, w;};
		struct {float r, g, b, a;};
	};
};

class UColor {
public:
	__device__ __host__ UColor(){}
	__device__ __host__ UColor(float4 c) {
		c4 = make_uchar4(c.x * 255, c.y * 255, c.z * 255, c.w * 255);
	}

	union {
		float c1;
		uchar4 c4;
	};
};

class Color {
public:
	__device__ __host__ Color(){ c = make_float4(0, 0, 0, 0); }
	__device__ __host__ Color(float r, float g, float b, float a) {
		c = make_float4(r, g, b, a);
	}

	__device__ __host__ Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
		c = make_float4(r/255.0f, g/255.0f, b/255.0f, a/255.0f);
	}
	__device__ __host__ uchar4 toURGBA() const {
		UColor color(c);
		return color.c4;
	}

	__device__ __host__ float toFloat() const{
		UColor color(c);
		return color.c1;
	}

	float4 c;
};

class Camera {
    float3 pos;
    float3 up, dir;
    float f;
    float w, h;
};

__device__ struct Light {
};

__device__ struct Hit {
	Color c;
	float t;
};

__device__ struct Shape {
};
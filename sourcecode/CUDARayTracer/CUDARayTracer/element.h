#pragma once
#include <iostream>
#include <algorithm>
#include <string>
using namespace std;

class vec2 {
public:
	__device__ __host__ vec2():x(0), y(0){}
	__device__ __host__ vec2(const float2& v):data(v){}
	__device__ __host__ vec2(float x, float y):x(x), y(y){}
	__device__ __host__ vec2(const vec2& v):x(v.x),y(v.y){}
	__device__ __host__ vec2& operator=(const vec2& v){
		x = v.x; y = v.y;
		return (*this);
	}

	__device__ __host__ float dot(const vec2& v) const {
		return x * v.x + y * v.y;
	}

	__device__ __host__ float norm() const {
		return sqrtf(x * x + y * y);
	}

	__device__ __host__ float length() const {
		return sqrtf(x * x + y * y);
	}

	__device__ __host__ void normalize() {
		float n = norm();
		if( n != 0 ) {
			x /= n; y /= n;
		}
	}

	__device__ __host__ vec2 normalized() const {
		float n = norm();
		if( n != 0 ) return vec2(x/n, y/n);
		else return (*this);
	}

	// unitary operator
	__device__ __host__ vec2 operator-() const { return vec2(-x, -y); }

	// arithmetic ops, element wise ops
	__device__ __host__ vec2 operator+(const vec2& v) const { return vec2(x + v.x, y + v.y); }
	__device__ __host__ vec2 operator-(const vec2& v) const { return vec2(x - v.x, y - v.y); }
	__device__ __host__ vec2 operator*(const vec2& v) const { return vec2(x * v.x, y * v.y); }
	__device__ __host__ vec2 operator/(const vec2& v) const { return vec2(x / v.x, y / v.y); }

	__device__ __host__ vec2 operator+=(const vec2& v) { x += v.x; y += v.y; return (*this); }
	__device__ __host__ vec2 operator-=(const vec2& v) { x -= v.x; y -= v.y; return (*this); }
	__device__ __host__ vec2 operator*=(const vec2& v) { x *= v.x; y *= v.y; return (*this); }
	__device__ __host__ vec2 operator/=(const vec2& v) { x /= v.x; y /= v.y; return (*this); }

	__device__ __host__ vec2 operator+(float f) const { return vec2(x + f, y + f); }
	__device__ __host__ vec2 operator-(float f) const { return vec2(x - f, y - f); }
	__device__ __host__ vec2 operator*(float f) const { return vec2(x * f, y * f); }
	__device__ __host__ vec2 operator/(float f) const { return vec2(x / f, y / f); }

	__device__ __host__ vec2 operator+=(float f) { x += f; y += f; return (*this); }
	__device__ __host__ vec2 operator-=(float f) { x -= f; y -= f; return (*this); }
	__device__ __host__ vec2 operator*=(float f) { x *= f; y *= f; return (*this); }
	__device__ __host__ vec2 operator/=(float f) { x /= f; y /= f; return (*this); }

	friend __device__ __host__ vec2 operator+(float f, const vec2& v);
	friend __device__ __host__ vec2 operator-(float f, const vec2& v);
	friend __device__ __host__ vec2 operator*(float f, const vec2& v);
	friend __device__ __host__ vec2 operator/(float f, const vec2& v);

	union {
		float2 data;
		struct {float x, y;};
		struct {float r, g;};
	};
};

__device__ __host__ __inline__ vec2 operator+(float f, const vec2& v) { return vec2(v.x + f, v.y + f); }
__device__ __host__ __inline__ vec2 operator-(float f, const vec2& v) { return vec2(v.x - f, v.y - f); }
__device__ __host__ __inline__ vec2 operator*(float f, const vec2& v) { return vec2(v.x * f, v.y * f); }
__device__ __host__ __inline__ vec2 operator/(float f, const vec2& v) { return vec2(v.x / f, v.y / f); }


class vec3 {
public:
	__device__ __host__ vec3():x(0), y(0), z(0){}
	__device__ __host__ vec3(const float3& v):x(v.x), y(v.y), z(v.z){}
	__device__ __host__ vec3(float x, float y, float z):x(x), y(y), z(z){}
	__device__ __host__ vec3(const vec3& v):x(v.x),y(v.y), z(v.z){}
	__device__ __host__ vec3& operator=(const vec3& v){
		x = v.x; y = v.y; z = v.z;
		return (*this);
	}

	// cross product constructor
	__device__ __host__ vec3(const vec3& v1, const vec3& v2) {
		x = v1.y * v2.z - v1.z * v2.y;
		y = v1.z * v2.x - v1.x * v2.z;
		z = v1.x * v2.y - v1.y * v2.x;
	}

	// vector ops
	__device__ __host__ vec3 cross(const vec3& v) const {
		return vec3(
			y*v.z - z * v.y,
			z*v.x - x * v.z,
			x*v.y - y * v.x
			);
	}

	__device__ __host__ float dot(const vec3& v) const {
		return x * v.x + y * v.y + z * v.z;
	}

	__device__ __host__ float norm() const {
		return sqrtf(x * x + y * y + z * z);
	}
	
	__device__ __host__ float length() const {
		return sqrtf(x * x + y * y + z * z);
	}

	__device__ __host__ void normalize() {
		float n = norm();
		if( n != 0 ) {
			x /= n; y /= n; z /= n;
		}
	}

	__device__ __host__ vec3 normalized() const {
		float n = norm();
		if( n != 0 ) return vec3(x/n, y/n, z/n);
		else return (*this);
	}

	// unitary operator
	__device__ __host__ vec3 operator-() const { return vec3(-x, -y, -z); }

	// arithmetic ops, element wise ops
	__device__ __host__ vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	__device__ __host__ vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	__device__ __host__ vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	__device__ __host__ vec3 operator/(const vec3& v) const { return vec3(x / v.x, y / v.y, z / v.z); }

	__device__ __host__ vec3 operator+=(const vec3& v) { x += v.x, y += v.y, z += v.z; return (*this); }
	__device__ __host__ vec3 operator-=(const vec3& v) { x -= v.x, y -= v.y, z -= v.z; return (*this); }
	__device__ __host__ vec3 operator*=(const vec3& v) { x *= v.x, y *= v.y, z *= v.z; return (*this); }
	__device__ __host__ vec3 operator/=(const vec3& v) { x /= v.x, y /= v.y, z /= v.z; return (*this); }

	__device__ __host__ vec3 operator+(float f) const { return vec3(x + f, y + f, z + f); }
	__device__ __host__ vec3 operator-(float f) const { return vec3(x - f, y - f, z - f); }
	__device__ __host__ vec3 operator*(float f) const { return vec3(x * f, y * f, z * f); }
	__device__ __host__ vec3 operator/(float f) const { return vec3(x / f, y / f, z / f); }

	__device__ __host__ vec3 operator+=(float f) { x += f, y += f, z += f; return (*this); }
	__device__ __host__ vec3 operator-=(float f) { x -= f, y -= f, z -= f; return (*this); }
	__device__ __host__ vec3 operator*=(float f) { x *= f, y *= f, z *= f; return (*this); }
	__device__ __host__ vec3 operator/=(float f) { x /= f, y /= f, z /= f; return (*this); }

	friend __device__ __host__ vec3 operator+(float f, const vec3& v);
	friend __device__ __host__ vec3 operator-(float f, const vec3& v);
	friend __device__ __host__ vec3 operator*(float f, const vec3& v);
	friend __device__ __host__ vec3 operator/(float f, const vec3& v);

	friend istream& operator>>(istream& is, vec3& v);
	friend ostream& operator<<(ostream& os, const vec3& v);

	union {
		float3 data;
		struct {float x, y, z, w;};
		struct {float r, g, b, a;};
	};
};

__device__ __host__ __inline__ vec3 operator+(float f, const vec3& v) { return vec3(v.x + f, v.y + f, v.z + f); }
__device__ __host__ __inline__ vec3 operator-(float f, const vec3& v) { return vec3(v.x - f, v.y - f, v.z - f); }
__device__ __host__ __inline__ vec3 operator*(float f, const vec3& v) { return vec3(v.x * f, v.y * f, v.z * f); }
__device__ __host__ __inline__ vec3 operator/(float f, const vec3& v) { return vec3(v.x / f, v.y / f, v.z / f); }

__host__ __inline__ istream& operator>>(istream& is, vec3& v) {
	is >> v.x >> v.y >> v.z;
	return is;
}

__host__ __inline__ ostream& operator<<(ostream& os, vec3& v) {
	os << v.x << ' ' << v.y << ' ' << v.z;
	return os;
}

class vec4 {
public:
	__device__ __host__ vec4(){}
	__device__ __host__ vec4(const float4& v):data(v){}
	__device__ __host__ vec4(float x, float y, float z, float w):x(x), y(y), z(z), w(w){}
	__device__ __host__ vec4(const vec4& v):x(v.x), y(v.y), z(v.z), w(v.w){}
	__device__ __host__ vec4(const vec3& v, float a):x(v.x), y(v.y), z(v.z), w(a){}
	__device__ __host__ vec4& operator=(const vec4& v){
		data = v.data;
		return (*this);
	}

	__device__ __host__ float norm() const {
		return sqrtf(x * x + y * y + z * z + w * w);
	}

	__device__ __host__ float length() const {
		return sqrtf(x * x + y * y + z * z + w * w);
	}
	
	__device__ __host__ void normalize() {
		float n = norm();
		if( n != 0 ) {
			x /= n; y /= n; z /= n; w /= n;
		}
	}

	// unitary operator
	__device__ __host__ vec4 operator-() const { return vec4(-x, -y, -z, -w); }

	// arithmetic ops, element wise ops
	__device__ __host__ vec4 operator+(const vec4& v) const { return vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
	__device__ __host__ vec4 operator-(const vec4& v) const { return vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
	__device__ __host__ vec4 operator*(const vec4& v) const { return vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
	__device__ __host__ vec4 operator/(const vec4& v) const { return vec4(x / v.x, y / v.y, z / v.z, w / v.w); }

	__device__ __host__ vec4 operator+(const vec4& v) { x += v.x, y += v.y, z += v.z, w += v.w; return (*this); }
	__device__ __host__ vec4 operator-(const vec4& v) { x -= v.x, y -= v.y, z -= v.z, w -= v.w; return (*this); }
	__device__ __host__ vec4 operator*(const vec4& v) { x *= v.x, y *= v.y, z *= v.z, w *= v.w; return (*this); }
	__device__ __host__ vec4 operator/(const vec4& v) { x /= v.x, y /= v.y, z /= v.z, w /= v.w; return (*this); }

	__device__ __host__ vec4 operator+(float f) const { return vec4(x + f, y + f, z + f, w + f); }
	__device__ __host__ vec4 operator-(float f) const { return vec4(x - f, y - f, z - f, w - f); }
	__device__ __host__ vec4 operator*(float f) const { return vec4(x * f, y * f, z * f, w * f); }
	__device__ __host__ vec4 operator/(float f) const { return vec4(x / f, y / f, z / f, w / f); }

	__device__ __host__ vec4 operator+=(float f) { x += f, y += f, z += f, w += f; return (*this); }
	__device__ __host__ vec4 operator-=(float f) { x -= f, y -= f, z -= f, w -= f; return (*this); }
	__device__ __host__ vec4 operator*=(float f) { x *= f, y *= f, z *= f, w *= f; return (*this); }
	__device__ __host__ vec4 operator/=(float f) { x /= f, y /= f, z /= f, w /= f; return (*this); }

	friend __device__ __host__ vec4 operator+(float f, const vec4& v);
	friend __device__ __host__ vec4 operator-(float f, const vec4& v);
	friend __device__ __host__ vec4 operator*(float f, const vec4& v);
	friend __device__ __host__ vec4 operator/(float f, const vec4& v);

	union {
		float4 data;
		struct {float x, y, z, w;};
		struct {float r, g, b, a;};
	};
};

__device__ __host__ __inline__ vec4 operator+(float f, const vec4& v) { return vec4(v.x + f, v.y + f, v.z + f, v.w + f); }
__device__ __host__ __inline__ vec4 operator-(float f, const vec4& v) { return vec4(v.x - f, v.y - f, v.z - f, v.w - f); }
__device__ __host__ __inline__ vec4 operator*(float f, const vec4& v) { return vec4(v.x * f, v.y * f, v.z * f, v.w * f); }
__device__ __host__ __inline__ vec4 operator/(float f, const vec4& v) { return vec4(v.x / f, v.y / f, v.z / f, v.w / f); }


class mat3 {
public:
	__device__ __host__ mat3(){}
	__device__ __host__ mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22) {
		elem[0] = m00; elem[1] = m01; elem[2] = m02;
		elem[3] = m10; elem[4] = m11; elem[5] = m12;
		elem[6] = m20; elem[7] = m21; elem[8] = m22;
	}
	__device__ __host__ mat3(const mat3& m){
		elem[0] = m.elem[0]; elem[1] = m.elem[1]; elem[2] = m.elem[2];
		elem[3] = m.elem[3]; elem[4] = m.elem[4]; elem[5] = m.elem[5];
		elem[6] = m.elem[6]; elem[7] = m.elem[7]; elem[8] = m.elem[8];
	}
	__device__ __host__ mat3(float *m) {
		elem[0] = m[0]; elem[1] = m[1]; elem[2] = m[2];
		elem[3] = m[3]; elem[4] = m[4]; elem[5] = m[5];
		elem[6] = m[6]; elem[7] = m[7]; elem[8] = m[8];
	}

	__device__ __host__ mat3(float3 row0, float3 row1, float3 row2) {
		elem[0] = row0.x; elem[1] = row0.y; elem[2] = row0.z;
		elem[3] = row1.x; elem[4] = row1.y; elem[5] = row1.z;
		elem[6] = row2.x; elem[7] = row2.y; elem[8] = row2.z;
	}

	__device__ __host__ mat3& operator=(const mat3& m) {
		elem[0] = m.elem[0]; elem[1] = m.elem[1]; elem[2] = m.elem[2];
		elem[3] = m.elem[3]; elem[4] = m.elem[4]; elem[5] = m.elem[5];
		elem[6] = m.elem[6]; elem[7] = m.elem[7]; elem[8] = m.elem[8];
		return (*this);
	}

	__device__ __host__ static mat3 zero() {
		return mat3(0, 0, 0, 0, 0, 0, 0, 0, 0);
	}
	__device__ __host__ static mat3 identity() {
		return mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);
	}

	__device__ __host__ static mat3 scaling(float sx, float sy, float sz) {
		return mat3(sx, 0, 0, 0, sy, 0, 0, 0, sz);
	}

	__device__ __host__ static mat3 rotation_x(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			1.0,	0.0,		0.0,
			0.0,	cosTheta,	-sinTheta,
			0.0,	sinTheta,	cosTheta
			);
	}

	__device__ __host__ static mat3 rotation_y(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			cosTheta,	0.0,	sinTheta,
			0.0,		1.0,	0.0,
			-sinTheta,	0.0,	cosTheta
			);
	}

	__device__ __host__ static mat3 rotation_z(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			cosTheta,	-sinTheta,	0.0,
			sinTheta,	cosTheta,	0.0,
			0.0,		0.0,		1.0
			);
	}

	__device__ __host__ static mat3 rotation_dx(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			1.0,	0.0,		0.0,
			0.0,	-sinTheta,	-cosTheta,
			0.0,	cosTheta,	-sinTheta
			);
	}

	__device__ __host__ static mat3 rotation_dy(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			-sinTheta,	0.0,	cosTheta,
			0.0,		1.0,	0.0,
			-cosTheta,	0.0,	-sinTheta
			);
	}

	__device__ __host__ static mat3 rotation_dz(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			-sinTheta,	-cosTheta,	0.0,
			cosTheta,	-sinTheta,	0.0,
			0.0,		0.0,		1.0
			);
	}

	__device__ __host__ static mat3 rotation(float rx, float ry, float rz) {
		mat3 mat = rotation_z(rz);
		mat *= rotation_y(ry);
		mat *= rotation_x(rx);
		return mat;
	}

	__device__ __host__ static void jacobian(
		float rx, float ry, float rz,
		mat3& Jx, mat3& Jy, mat3& Jz
		) {
			mat3 Rx = rotation_x(rx);
			mat3 Ry = rotation_y(ry);
			mat3 Rz = rotation_z(rz);

			Jx = Rz * Ry * rotation_dx(rx);
			Jy = Rz * rotation_dy(ry) * Rx;
			Jz = rotation_dz(rz) * Ry * Rx;
	}

	__device__ __host__ mat3 operator-() const {
		return mat3(-elem[0], -elem[1], -elem[2], -elem[3], -elem[4], -elem[5], 
			-elem[6], -elem[7], -elem[8]);
	}

	__device__ __host__ mat3 operator+(const mat3& m) const {
		return mat3(
			elem[0]+m(0), elem[1]+m(1), elem[2]+m(2),
			elem[3]+m(3), elem[4]+m(4), elem[5]+m(5),
			elem[6]+m(6), elem[7]+m(7), elem[8]+m(8)
			);
	}

	__device__ __host__ mat3 operator-(const mat3& m) {
		return mat3(
			elem[0]-m(0), elem[1]-m(1), elem[2]-m(2),
			elem[3]-m(3), elem[4]-m(4), elem[5]-m(5),
			elem[6]-m(6), elem[7]-m(7), elem[8]-m(8)
			);
	}

	__device__ __host__ mat3 operator*(const mat3& m) {
		mat3 res;
		res(0) = elem[0] * m(0) + elem[1] * m(3) + elem[2] * m(6);
		res(1) = elem[0] * m(1) + elem[1] * m(4) + elem[2] * m(7);
		res(2) = elem[0] * m(2) + elem[1] * m(5) + elem[2] * m(8);

		res(3) = elem[3] * m(0) + elem[4] * m(3) + elem[5] * m(6);
		res(4) = elem[3] * m(1) + elem[4] * m(4) + elem[5] * m(7);
		res(5) = elem[3] * m(2) + elem[4] * m(5) + elem[5] * m(8);

		res(6) = elem[6] * m(0) + elem[7] * m(3) + elem[8] * m(6);
		res(7) = elem[6] * m(1) + elem[7] * m(4) + elem[8] * m(7);
		res(8) = elem[6] * m(2) + elem[7] * m(5) + elem[8] * m(8);

		return res;
	}

	__device__ __host__ mat3 operator*=(const mat3& m) {
		(*this) = (*this)*m;
		return (*this);
	}

	__device__ __host__ vec3 operator*(const vec3& v) {
		return vec3(
			elem[0] * v.x + elem[1] * v.y + elem[2] * v.z,
			elem[3] * v.x + elem[4] * v.y + elem[5] * v.z,
			elem[6] * v.x + elem[7] * v.y + elem[8] * v.z
			);
	}

	__device__ __host__ float3 operator*(const float3& v) {
		return make_float3(
			elem[0] * v.x + elem[1] * v.y + elem[2] * v.z,
			elem[3] * v.x + elem[4] * v.y + elem[5] * v.z,
			elem[6] * v.x + elem[7] * v.y + elem[8] * v.z
			);
	}

	__device__ __host__ mat3 operator*(float f) {
		return mat3(
			elem[0] * f, elem[1] * f, elem[2] * f,
			elem[3] * f, elem[4] * f, elem[5] * f,
			elem[6] * f, elem[7] * f, elem[8] * f
			);
	}

	__device__ __host__ friend mat3 operator*(float f, const mat3& m);

	__device__ __host__ mat3 trans() const {
		return mat3(
			elem[0], elem[3], elem[6],
			elem[1], elem[4], elem[7],
			elem[2], elem[5], elem[8]
			);
	}

	__device__ __host__ float det() const {
		return elem[0]*(elem[4]*elem[8]-elem[5]*elem[7])
			  -elem[1]*(elem[3]*elem[8]-elem[5]*elem[6])
			  +elem[2]*(elem[3]*elem[7]-elem[4]*elem[6]);
	}

	__device__ __host__ mat3 inv() const {
		float D = det();
		if( D == 0 ) return mat3::zero();
		else {
			float invD = 1.0f / D;
			mat3 res;
			res(0, 0) = (elem[4] * elem[8] - elem[7] * elem[5]) * invD;
			res(0, 1) = (elem[7] * elem[2] - elem[1] * elem[8]) * invD;
			res(0, 2) = (elem[1] * elem[5] - elem[4] * elem[2]) * invD;

			res(1, 0) = (elem[5] * elem[6] - elem[3] * elem[8]) * invD;
			res(1, 1) = (elem[8] * elem[0] - elem[6] * elem[2]) * invD;
			res(1, 2) = (elem[2] * elem[3] - elem[0] * elem[5]) * invD;

			res(2, 0) = (elem[3] * elem[7] - elem[4] * elem[6]) * invD;
			res(2, 1) = (elem[6] * elem[1] - elem[7] * elem[0]) * invD;
			res(2, 2) = (elem[0] * elem[4] - elem[1] * elem[3]) * invD;

			return res;
		}
	}

	__device__ __host__ float operator()(int idx) const { return elem[idx]; }
	__device__ __host__ float& operator()(int idx) { return elem[idx]; }

	__device__ __host__ float operator()(int r, int c) const { return elem[r*3+c]; }
	__device__ __host__ float& operator()(int r, int c) { return elem[r*3+c]; }

	float elem[9];
};

__device__ __host__ __inline__ mat3 operator*(float f, const mat3& m) {
	return mat3(
		m.elem[0] * f, m.elem[1] * f, m.elem[2] * f,
		m.elem[3] * f, m.elem[4] * f, m.elem[5] * f,
		m.elem[6] * f, m.elem[7] * f, m.elem[8] * f
		);
}

class mat4 {
public:
	__device__ __host__ mat4(){
		for(int i=0;i<16;i++) elem[i] = 0;
	}
	__device__ __host__ mat4(const mat4& m){
		for(int i=0;i<16;i++) elem[i] = m.elem[i];
	}
	__device__ __host__ mat4(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33
		)
	{
		elem[ 0] = m00; elem[ 1] = m01; elem[ 2] = m02; elem[ 3] = m03;
		elem[ 4] = m10; elem[ 5] = m11; elem[ 6] = m12; elem[ 7] = m13;
		elem[ 8] = m20; elem[ 9] = m21; elem[10] = m22; elem[11] = m23;
		elem[12] = m30; elem[13] = m31; elem[14] = m32; elem[15] = m33;
	}
	__device__ __host__ mat4(float *m) {
		elem[0] = m[0]; elem[1] = m[1]; elem[2] = m[2]; elem[3] = m[3];
		elem[4] = m[4]; elem[5] = m[5];	elem[6] = m[6]; elem[7] = m[7]; 
		elem[8] = m[8]; elem[9] = m[9]; elem[10] = m[10]; elem[11] = m[11];
		elem[12] = m[12]; elem[13] = m[13]; elem[14] = m[14]; elem[15] = m[15];
	}
	__device__ __host__ ~mat4(){}

	__device__ __host__ mat4& operator=(const mat4& m){
		for(int i=0;i<16;i++) elem[i] = m.elem[i];
		return (*this);
	}

	__device__ __host__ static mat4 zero(){ return mat4(); }
	__device__ __host__ static mat4 identity(){ 
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
			); 
	}

	__device__ __host__ mat4 trans() const {
		return mat4(
			elem[0], elem[4], elem[8], elem[12],
			elem[1], elem[5], elem[9], elem[13],
			elem[2], elem[6], elem[10], elem[14],
			elem[3], elem[7], elem[11], elem[15]
			);
	}

	__device__ __host__ mat4 inv() const {
		const float* pm = &(elem[0]);
		float S0 = pm[0] * pm[5] - pm[1] * pm[4];
		float S1 = pm[0] * pm[6] - pm[2] * pm[4];
		float S2 = pm[0] * pm[7] - pm[3] * pm[4];
		float S3 = pm[1] * pm[6] - pm[2] * pm[5];
		float S4 = pm[1] * pm[7] - pm[3] * pm[5];
		float S5 = pm[2] * pm[7] - pm[3] * pm[6];

		float C5 = pm[10] * pm[15] - pm[11] * pm[14];
		float C4 = pm[9] * pm[15] - pm[11] * pm[13];
		float C3 = pm[9] * pm[14] - pm[10] * pm[13];
		float C2 = pm[8] * pm[15] - pm[11] * pm[12];
		float C1 = pm[8] * pm[14] - pm[10] * pm[12];
		float C0 = pm[8] * pm[13] - pm[9] * pm[12];

		// If determinant equals 0, there is no inverse
		float D = S0 * C5 - S1 * C4 + S2 * C3 + S3 * C2 - S4 * C1 + S5 * C0;
		if(fabs(D) <= 1e-8) return mat4();

		float dinv = 1.0f / D;
		// Compute adjugate matrix
		mat4 mat(
			pm[5] * C5  - pm[6] * C4  + pm[7] * C3,  -pm[1] * C5 + pm[2] * C4  - pm[3] * C3,
			pm[13] * S5 - pm[14] * S4 + pm[15] * S3, -pm[9] * S5 + pm[10] * S4 - pm[11] * S3,

			-pm[4] * C5  + pm[6] * C2  - pm[7] * C1,   pm[0] * C5 - pm[2] * C2  + pm[3] * C1,
			-pm[12] * S5 + pm[14] * S2 - pm[15] * S1,  pm[8] * S5 - pm[10] * S2 + pm[11] * S1,

			pm[4] * C4  - pm[5] * C2  + pm[7] * C0,  -pm[0] * C4 + pm[1] * C2  - pm[3] * C0,
			pm[12] * S4 - pm[13] * S2 + pm[15] * S0, -pm[8] * S4 + pm[9] * S2  - pm[11] * S0,

			-pm[4] * C3  + pm[5] * C1  - pm[6] * C0,   pm[0] * C3 - pm[1] * C1  + pm[2] * C0,
			-pm[12] * S3 + pm[13] * S1 - pm[14] * S0,  pm[8] * S3 - pm[9] * S1  + pm[10] * S0 
			);

		mat *= dinv;

		return mat;
	}

	__device__ __host__ float det() const {
		const float* pm = &(elem[0]);
		float S0 = pm[0] * pm[5] - pm[1] * pm[4];
		float S1 = pm[0] * pm[6] - pm[2] * pm[4];
		float S2 = pm[0] * pm[7] - pm[3] * pm[4];
		float S3 = pm[1] * pm[6] - pm[2] * pm[5];
		float S4 = pm[1] * pm[7] - pm[3] * pm[5];
		float S5 = pm[2] * pm[7] - pm[3] * pm[6];

		float C5 = pm[10] * pm[15] - pm[11] * pm[14];
		float C4 = pm[9] * pm[15] - pm[11] * pm[13];
		float C3 = pm[9] * pm[14] - pm[10] * pm[13];
		float C2 = pm[8] * pm[15] - pm[11] * pm[12];
		float C1 = pm[8] * pm[14] - pm[10] * pm[12];
		float C0 = pm[8] * pm[13] - pm[9] * pm[12];

		return S0 * C5 - S1 * C4 + S2 * C3 + S3 * C2 - S4 * C1 + S5 * C0;
	}

	__device__ __host__ mat4 operator+(const mat4& m) {
		return mat4(
			elem[0] + m.elem[0], elem[1] + m.elem[1], elem[2] + m.elem[2], elem[3] + m.elem[3],
			elem[4] + m.elem[4], elem[5] + m.elem[5], elem[6] + m.elem[6], elem[7] + m.elem[7],
			elem[8] + m.elem[8], elem[9] + m.elem[9], elem[10] + m.elem[10], elem[11] + m.elem[11],
			elem[12] + m.elem[12], elem[13] + m.elem[13], elem[14] + m.elem[14], elem[15] + m.elem[15]
		);
	}
	__device__ __host__ mat4 operator-(const mat4& m) {
		return mat4(
			elem[0] - m.elem[0], elem[1] - m.elem[1], elem[2] - m.elem[2], elem[3] - m.elem[3],
			elem[4] - m.elem[4], elem[5] - m.elem[5], elem[6] - m.elem[6], elem[7] + m.elem[7],
			elem[8] - m.elem[8], elem[9] - m.elem[9], elem[10] - m.elem[10], elem[11] - m.elem[11],
			elem[12] - m.elem[12], elem[13] - m.elem[13], elem[14] - m.elem[14], elem[15] - m.elem[15]
		);
	}
	__device__ __host__ mat4 operator*(const mat4& m) {
		mat4 res;
		res(0)  = elem[0] * m(0) + elem[1] * m(4) + elem[2] * m( 8) + elem[3] * m(12);
		res(1)  = elem[0] * m(1) + elem[1] * m(5) + elem[2] * m( 9) + elem[3] * m(13);
		res(2)  = elem[0] * m(2) + elem[1] * m(6) + elem[2] * m(10) + elem[3] * m(14);
		res(3)  = elem[0] * m(3) + elem[1] * m(7) + elem[2] * m(11) + elem[3] * m(15);

		res(4)  = elem[4] * m(0, 0) + elem[5] * m(1, 0) + elem[6] * m(2, 0) + elem[7] * m(3, 0);
		res(5)  = elem[4] * m(0, 1) + elem[5] * m(1, 1) + elem[6] * m(2, 1) + elem[7] * m(3, 1);
		res(6)  = elem[4] * m(0, 2) + elem[5] * m(1, 2) + elem[6] * m(2, 2) + elem[7] * m(3, 2);
		res(7)  = elem[4] * m(0, 3) + elem[5] * m(1, 3) + elem[6] * m(2, 3) + elem[7] * m(3, 3);

		res(8)  = elem[8] * m(0, 0) + elem[9] * m(1, 0) + elem[10] * m(2, 0) + elem[11] * m(3, 0);
		res(9)  = elem[8] * m(0, 1) + elem[9] * m(1, 1) + elem[10] * m(2, 1) + elem[11] * m(3, 1);
		res(10) = elem[8] * m(0, 2) + elem[9] * m(1, 2) + elem[10] * m(2, 2) + elem[11] * m(3, 2);
		res(11) = elem[8] * m(0, 3) + elem[9] * m(1, 3) + elem[10] * m(2, 3) + elem[11] * m(3, 3);

		res(12) = elem[12] * m(0, 0) + elem[13] * m(1, 0) + elem[14] * m(2, 0) + elem[15] * m(3, 0);
		res(13) = elem[12] * m(0, 1) + elem[13] * m(1, 1) + elem[14] * m(2, 1) + elem[15] * m(3, 1);
		res(14) = elem[12] * m(0, 2) + elem[13] * m(1, 2) + elem[14] * m(2, 2) + elem[15] * m(3, 2);
		res(15) = elem[12] * m(0, 3) + elem[13] * m(1, 3) + elem[14] * m(2, 3) + elem[15] * m(3, 3);
		return res;
	}

	__device__ __host__ vec4 operator*(const vec4& v) {
		vec4 res;
		res.x = v.x * elem[0] + v.y * elem[1] + v.z * elem[2] + v.w * elem[3];
		res.y = v.x * elem[4] + v.y * elem[5] + v.z * elem[6] + v.w * elem[7];
		res.z = v.x * elem[8] + v.y * elem[9] + v.z * elem[10] + v.w * elem[11];
		res.w = v.x * elem[12] + v.y * elem[13] + v.z * elem[14] + v.w * elem[15];
		return res;
	}

	__device__ __host__ vec3 operator*(const vec3& v) {
		vec4 vv(v, 1.0);
		vv = (*this) * vv;
		return vec3(vv.x, vv.y, vv.z);
	}

	__device__ __host__ mat4 operator*(float f) {
		return mat4(
			elem[0]*f, elem[1]*f, elem[2]*f, elem[3]*f,
			elem[4]*f, elem[5]*f, elem[6]*f, elem[7]*f,
			elem[8]*f, elem[9]*f, elem[10]*f, elem[11]*f,
			elem[12]*f, elem[13]*f, elem[14]*f, elem[15]*f
		);
	}
	__device__ __host__ mat4 operator+=(const mat4& m) {
		elem[0] += m.elem[0]; elem[1] += m.elem[1]; elem[2] += m.elem[2]; elem[3] += m.elem[3];
		elem[4] += m.elem[4]; elem[5] += m.elem[5]; elem[6] += m.elem[6]; elem[7] += m.elem[7];
		elem[8] += m.elem[8]; elem[9] += m.elem[9]; elem[10] += m.elem[10]; elem[11] += m.elem[11];
		elem[12] += m.elem[12]; elem[13] += m.elem[13]; elem[14] += m.elem[14]; elem[15] += m.elem[15];
		return (*this);
	}
	__device__ __host__ mat4 operator-=(const mat4& m) {
		elem[0] -= m.elem[0]; elem[1] -= m.elem[1]; elem[2] -= m.elem[2]; elem[3] -= m.elem[3];
		elem[4] -= m.elem[4]; elem[5] -= m.elem[5]; elem[6] -= m.elem[6]; elem[7] -= m.elem[7];
		elem[8] -= m.elem[8]; elem[9] -= m.elem[9]; elem[10] -= m.elem[10]; elem[11] -= m.elem[11];
		elem[12] -= m.elem[12]; elem[13] -= m.elem[13]; elem[14] -= m.elem[14]; elem[15] -= m.elem[15];
		return (*this);
	}

	__device__ __host__ mat4 operator*=(const mat4& m) {
		mat4 res;
		res(0)  = elem[0] * m(0) + elem[1] * m(4) + elem[2] * m( 8) + elem[3] * m(12);
		res(1)  = elem[0] * m(1) + elem[1] * m(5) + elem[2] * m( 9) + elem[3] * m(13);
		res(2)  = elem[0] * m(2) + elem[1] * m(6) + elem[2] * m(10) + elem[3] * m(14);
		res(3)  = elem[0] * m(3) + elem[1] * m(7) + elem[2] * m(11) + elem[3] * m(15);

		res(4)  = elem[4] * m(0, 0) + elem[5] * m(1, 0) + elem[6] * m(2, 0) + elem[7] * m(3, 0);
		res(5)  = elem[4] * m(0, 1) + elem[5] * m(1, 1) + elem[6] * m(2, 1) + elem[7] * m(3, 1);
		res(6)  = elem[4] * m(0, 2) + elem[5] * m(1, 2) + elem[6] * m(2, 2) + elem[7] * m(3, 2);
		res(7)  = elem[4] * m(0, 3) + elem[5] * m(1, 3) + elem[6] * m(2, 3) + elem[7] * m(3, 3);

		res(8)  = elem[8] * m(0, 0) + elem[9] * m(1, 0) + elem[10] * m(2, 0) + elem[11] * m(3, 0);
		res(9)  = elem[8] * m(0, 1) + elem[9] * m(1, 1) + elem[10] * m(2, 1) + elem[11] * m(3, 1);
		res(10) = elem[8] * m(0, 2) + elem[9] * m(1, 2) + elem[10] * m(2, 2) + elem[11] * m(3, 2);
		res(11) = elem[8] * m(0, 3) + elem[9] * m(1, 3) + elem[10] * m(2, 3) + elem[11] * m(3, 3);

		res(12) = elem[12] * m(0, 0) + elem[13] * m(1, 0) + elem[14] * m(2, 0) + elem[15] * m(3, 0);
		res(13) = elem[12] * m(0, 1) + elem[13] * m(1, 1) + elem[14] * m(2, 1) + elem[15] * m(3, 1);
		res(14) = elem[12] * m(0, 2) + elem[13] * m(1, 2) + elem[14] * m(2, 2) + elem[15] * m(3, 2);
		res(15) = elem[12] * m(0, 3) + elem[13] * m(1, 3) + elem[14] * m(2, 3) + elem[15] * m(3, 3);
		(*this) = res;
		return (*this);
	}

	__device__ __host__ mat4 operator*=(float f) {
		elem[0] *= f; elem[1] *= f; elem[2] *= f; elem[3] *= f;
		elem[4] *= f; elem[5] *= f; elem[6] *= f; elem[7] *= f;
		elem[8] *= f; elem[9] *= f; elem[10] *= f; elem[11] *= f;
		elem[12] *= f; elem[13] *= f; elem[14] *= f; elem[15] *= f;
		return (*this);
	}

	__device__ __host__ float operator()(int idx) const { return elem[idx]; }
	__device__ __host__ float& operator()(int idx) { return elem[idx]; }

	__device__ __host__ float operator()(int r, int c) const { return elem[r*4+c]; }
	__device__ __host__ float& operator()(int r, int c) { return elem[r*4+c]; }

	float elem[16];
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
	__device__ __host__ Color(){ c.data = make_float4(0, 0, 0, 0); }
	__device__ __host__ Color(float r, float g, float b, float a) {
		c.data = make_float4(r, g, b, a);
	}

	__device__ __host__ Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
		c.data = make_float4(r/255.0f, g/255.0f, b/255.0f, a/255.0f);
	}
	__device__ __host__ uchar4 toURGBA() const {
		UColor color(c.data);
		return color.c4;
	}

	__device__ __host__ float toFloat() const{
		UColor color(c.data);
		return color.c1;
	}

	vec4 c;
};

struct TextureObject {
	enum TextureType {
		Julia2D = 10001,
		Julia,
		Perlin2D,
		Perlin,
		Chessboard2D,
		Chessboard,
		Marble,
		WoodGrain,
		Image
	};

	static TextureType parseType(const string& str) {
		if( str == "julia2d" ) return Julia2D;
		else if( str == "julia" ) return Julia;
		else if( str == "perlin2d" ) return Perlin2D;
		else if( str == "perlin" ) return Perlin;
		else if( str == "chessboard2d" ) return Chessboard2D;
		else if( str == "chessboard" ) return Chessboard;
		else if( str == "marble" ) return Marble;
		else if( str == "woodgrain" ) return WoodGrain;
		else return Image;
	}

	bool isHDR;
	float4 *addrf;
	uchar4 *addr;
	int2 size;
};
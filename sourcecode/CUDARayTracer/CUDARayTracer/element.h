#pragma once

class vec2 {
public:
	__device__ __host__ vec2():x(0), y(0){}
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

	__device__ __host__ vec2 operator+(float f) const { return vec2(x + f, y + f); }
	__device__ __host__ vec2 operator-(float f) const { return vec2(x - f, y - f); }
	__device__ __host__ vec2 operator*(float f) const { return vec2(x * f, y * f); }
	__device__ __host__ vec2 operator/(float f) const { return vec2(x / f, y / f); }

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

	__device__ __host__ vec3 operator+(float f) const { return vec3(x + f, y + f, z + f); }
	__device__ __host__ vec3 operator-(float f) const { return vec3(x - f, y - f, z - f); }
	__device__ __host__ vec3 operator*(float f) const { return vec3(x * f, y * f, z * f); }
	__device__ __host__ vec3 operator/(float f) const { return vec3(x / f, y / f, z / f); }

	friend __device__ __host__ vec3 operator+(float f, const vec3& v);
	friend __device__ __host__ vec3 operator-(float f, const vec3& v);
	friend __device__ __host__ vec3 operator*(float f, const vec3& v);
	friend __device__ __host__ vec3 operator/(float f, const vec3& v);

	union {
		float3 data;
		struct {float x, y, z;};
		struct {float r, g, b;};
	};
};

__device__ __host__ __inline__ vec3 operator+(float f, const vec3& v) { return vec3(v.x + f, v.y + f, v.z + f); }
__device__ __host__ __inline__ vec3 operator-(float f, const vec3& v) { return vec3(v.x - f, v.y - f, v.z - f); }
__device__ __host__ __inline__ vec3 operator*(float f, const vec3& v) { return vec3(v.x * f, v.y * f, v.z * f); }
__device__ __host__ __inline__ vec3 operator/(float f, const vec3& v) { return vec3(v.x / f, v.y / f, v.z / f); }

class vec4 {
public:
	__device__ __host__ vec4(){}
	__device__ __host__ vec4(float x, float y, float z, float w):x(x), y(y), z(z), w(w){}
	__device__ __host__ vec4(const vec4& v):x(v.x), y(v.y), z(v.z), w(v.w){}
	__device__ __host__ vec4(const vec3& v, float a):x(v.x), y(v.y), z(v.z), w(a){}
	__device__ __host__ vec4& operator=(const vec4& v){
		data = v.data;
		return (*this);
	}

	// unitary operator
	__device__ __host__ vec4 operator-() const { return vec4(-x, -y, -z, -w); }

	// arithmetic ops, element wise ops
	__device__ __host__ vec4 operator+(const vec4& v) const { return vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
	__device__ __host__ vec4 operator-(const vec4& v) const { return vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
	__device__ __host__ vec4 operator*(const vec4& v) const { return vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
	__device__ __host__ vec4 operator/(const vec4& v) const { return vec4(x / v.x, y / v.y, z / v.z, w / v.w); }

	__device__ __host__ vec4 operator+(float f) const { return vec4(x + f, y + f, z + f, w + f); }
	__device__ __host__ vec4 operator-(float f) const { return vec4(x - f, y - f, z - f, w - f); }
	__device__ __host__ vec4 operator*(float f) const { return vec4(x * f, y * f, z * f, w * f); }
	__device__ __host__ vec4 operator/(float f) const { return vec4(x / f, y / f, z / f, w / f); }

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

	__device__ __host__ mat3 operator+(const mat3& m) {
		return mat3(
			elem[0]+m(0), elem[1]+m(1), elem[2]+m(2),
			elem[3]+m(3), elem[4]+m(4), elem[5]+m(5),
			elem[6]+m(6), elem[7]+m(7), elem[8]+m(8)
			);
	}

	__device__ __host__ mat3 operator-(const mat3& m) {
		return mat3(
			elem[0]+m(0), elem[1]+m(1), elem[2]+m(2),
			elem[3]+m(3), elem[4]+m(4), elem[5]+m(5),
			elem[6]+m(6), elem[7]+m(7), elem[8]+m(8)
			);
	}

	__device__ __host__ mat3 operator*(const mat3& m) {
		mat3 res;
		res(0) = elem[0] * m(0) + elem[1] * m(3) + elem[2] * m(6);
		res(1) = elem[0] * m(1) + elem[1] * m(4) + elem[2] * m(7);
		res(2) = elem[0] * m(2) + elem[1] * m(5) + elem[2] * m(8);

		res(3) = elem[0] * m(0) + elem[1] * m(3) + elem[2] * m(6);
		res(4) = elem[0] * m(1) + elem[1] * m(4) + elem[2] * m(7);
		res(5) = elem[0] * m(2) + elem[1] * m(5) + elem[2] * m(8);

		res(6) = elem[0] * m(0) + elem[1] * m(3) + elem[2] * m(6);
		res(7) = elem[0] * m(1) + elem[1] * m(4) + elem[2] * m(7);
		res(8) = elem[0] * m(2) + elem[1] * m(5) + elem[2] * m(8);

		return res;
	}

	__device__ __host__ vec3 operator*(const vec3& v) {
		return vec3(
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
	__device__ __host__ mat4(){}
	__device__ __host__ mat4(const mat4& m){}

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
#pragma once

#include <thrust/random.h>
#include <cuda_runtime.h>
#include <helper_math.h> 

class PerlinNoise {
private:
	static const int B = 0x100;
	static const int BM = 0xff;
	static const int N = 0x1000;
	static const int NP = 12;
	static const int NM = 0xfff;

	int p[B+B+2];
	float g3[B+B+2][3];
	float g2[B+B+2][2];
	float g1[B+B+2];
	int start;

protected:
	__device__ void normalize2(float v[2])
	{
		float s;

		s = sqrt(v[0] * v[0] + v[1] * v[1]);
		v[0] = v[0] / s;
		v[1] = v[1] / s;
	}

	__device__ void normalize3(float v[3])
	{
		float s;

		s = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		v[0] = v[0] / s;
		v[1] = v[1] / s;
		v[2] = v[2] / s;
	}

public:
	__device__ void init() {
		thrust::default_random_engine rng(19557);
		thrust::uniform_int_distribution<int> u0max;

		int i, j, k;

		for (i = 0 ; i < B ; i++) {
			p[i] = i;

			g1[i] = (float)((u0max(rng) % (B + B)) - B) / B;

			for (j = 0 ; j < 2 ; j++)
				g2[i][j] = (float)((u0max(rng) % (B + B)) - B) / B;
			normalize2(g2[i]);

			for (j = 0 ; j < 3 ; j++)
				g3[i][j] = (float)((u0max(rng) % (B + B)) - B) / B;
			normalize3(g3[i]);
		}

		while (--i) {
			k = p[i];
			p[i] = p[j = u0max(rng) % B];
			p[j] = k;
		}

		for (i = 0 ; i < B + 2 ; i++) {
			p[B + i] = p[i];
			g1[B + i] = g1[i];
			for (j = 0 ; j < 2 ; j++)
				g2[B + i][j] = g2[i][j];
			for (j = 0 ; j < 3 ; j++)
				g3[B + i][j] = g3[i][j];
		}	
	}

	__device__ __forceinline__ float lerp(float t, float a, float b) {
		return a + t * (b - a);
	}

	__device__ __forceinline__ float s_curve(float t) { return t * t * (3. - 2. * t); }

	__device__ void setup(int i, int &b0, int &b1, float &r0, float &r1, float vec[]) {
		float t = vec[i] + N;
		b0 = ((int)t) & BM;
		b1 = (b0+1) & BM;
		r0 = t - (int)t;
		r1 = r0 - 1.;
	}

	__device__ float noise1(float arg) {
		int bx0, bx1;
		float rx0, rx1, sx, u, v, vec[1];

		vec[0] = arg;
		if (start) {
			start = 0;
			init();
		}
		setup(0, bx0, bx1, rx0, rx1, vec);

		sx = s_curve(rx0);

		u = rx0 * g1[ p[ bx0 ] ];
		v = rx1 * g1[ p[ bx1 ] ];

		return lerp(sx, u, v);
	}

	__device__ float noise2(float vec[2])
	{
		int bx0, bx1, by0, by1, b00, b10, b01, b11;
		float rx0, rx1, ry0, ry1, *q, sx, sy, a, b, t, u, v;
		register int i, j;

		if (start) {
			start = 0;
			init();
		}

		setup(0, bx0, bx1, rx0, rx1, vec);
		setup(1, by0, by1, ry0, ry1, vec);

		i = p[ bx0 ];
		j = p[ bx1 ];

		b00 = p[ i + by0 ];
		b10 = p[ j + by0 ];
		b01 = p[ i + by1 ];
		b11 = p[ j + by1 ];

		sx = s_curve(rx0);
		sy = s_curve(ry0);

	#define at2(rx,ry) ( rx * q[0] + ry * q[1] )

		q = g2[ b00 ] ; u = at2(rx0,ry0);
		q = g2[ b10 ] ; v = at2(rx1,ry0);
		a = lerp(sx, u, v);

		q = g2[ b01 ] ; u = at2(rx0,ry1);
		q = g2[ b11 ] ; v = at2(rx1,ry1);
		b = lerp(sx, u, v);

		return lerp(sy, a, b);
	}

	__device__ float noise3(float vec[3])
	{
		int bx0, bx1, by0, by1, bz0, bz1, b00, b10, b01, b11;
		float rx0, rx1, ry0, ry1, rz0, rz1, *q, sy, sz, a, b, c, d, t, u, v;
		register int i, j;

		if (start) {
			start = 0;
			init();
		}

		setup(0, bx0, bx1, rx0, rx1, vec);
		setup(1, by0, by1, ry0, ry1, vec);
		setup(2, bz0, bz1, rz0, rz1, vec);

		i = p[ bx0 ];
		j = p[ bx1 ];

		b00 = p[ i + by0 ];
		b10 = p[ j + by0 ];
		b01 = p[ i + by1 ];
		b11 = p[ j + by1 ];

		t  = s_curve(rx0);
		sy = s_curve(ry0);
		sz = s_curve(rz0);

	#define at3(rx,ry,rz) ( rx * q[0] + ry * q[1] + rz * q[2] )

		q = g3[ b00 + bz0 ] ; u = at3(rx0,ry0,rz0);
		q = g3[ b10 + bz0 ] ; v = at3(rx1,ry0,rz0);
		a = lerp(t, u, v);

		q = g3[ b01 + bz0 ] ; u = at3(rx0,ry1,rz0);
		q = g3[ b11 + bz0 ] ; v = at3(rx1,ry1,rz0);
		b = lerp(t, u, v);

		c = lerp(sy, a, b);

		q = g3[ b00 + bz1 ] ; u = at3(rx0,ry0,rz1);
		q = g3[ b10 + bz1 ] ; v = at3(rx1,ry0,rz1);
		a = lerp(t, u, v);

		q = g3[ b01 + bz1 ] ; u = at3(rx0,ry1,rz1);
		q = g3[ b11 + bz1 ] ; v = at3(rx1,ry1,rz1);
		b = lerp(t, u, v);

		d = lerp(sy, a, b);

		return lerp(sz, c, d);
	}
};
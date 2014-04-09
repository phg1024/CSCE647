#pragma once

#include "ray.h"
#include "props.h"

#include "helper_math.h"

struct PixelState {
	operator bool() const { return isActive; }
	int idx;			// pixel index: pix_y * imagesize.x + pix_x
	Ray ray;
	float3 accumulatedColor;
	float3 colormask;
	AbsorptionAndScatteringProp asprop;
	bool isActive;		// is the ray terminated
};

struct PixelActivivtyTester {
	PixelActivivtyTester(PixelState* pixels):pixels(pixels){}

	bool operator()(int idx) {
		return !(pixels[idx].isActive);
	}

	PixelState* pixels;
};
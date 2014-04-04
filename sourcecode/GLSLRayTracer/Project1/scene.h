#pragma once

#include "common.h"

#include "light.h"
#include "shape.h"
#include "camera.h"

struct Scene {
	Scene(){}

	Camera cam;
	vector<Shape> shapes;
	vector<Light> lights;
};
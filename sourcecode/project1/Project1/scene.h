#pragma once

#include "common.h"

#include "light.h"
#include "shape.h"
#include "camera.h"

struct Scene {
	Scene(){}

    string createInitializationSourceCode() const;
    string createShapesSourceCode() const;
    string createLightSourceCode() const;

	Camera cam;
	vector<Shape> shapes;
	vector<Light> lights;
};

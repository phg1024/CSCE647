#pragma once

#include "definitions.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;

class Scene
{
public:
	Scene(void);
	~Scene(void);

	bool load(const string& filename);
	bool save(const string& filename);

	const vector<Shape>& getShapes() const {return shapes;}
	const Camera& camera() const {return cam;}
	const vector<TextureObject>& getTextures() const { return texs; }

	const int getEnvironmentMap() const {
		return envmap;
	}
protected:
	void parse(const string& line);

private:
	Camera cam;
	vector<Shape> shapes;

	vector<TextureObject> texs;

	int envmap;
};

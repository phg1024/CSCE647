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
	const vector<Material>& getMaterials() const {return materials;}
	const vector<int>& getLights() const { 
		lights.clear();
		for(int i=0;i<shapes.size();i++) {
			if( shapes[i].material.t == Material::Emissive ) lights.push_back(i);
		}
		return lights;
	}
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
	vector<Material> materials;
	mutable vector<int> lights;
	vector<TextureObject> texs;
	map<string, int> materialMap;

	int envmap;
};

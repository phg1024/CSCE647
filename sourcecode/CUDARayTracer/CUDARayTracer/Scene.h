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

protected:
	void parse(const string& line);

private:
	Camera cam;
	vector<Shape> shapes;
};

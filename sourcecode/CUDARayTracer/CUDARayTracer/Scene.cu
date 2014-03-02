#include "Scene.h"
#include "utils.h"

Scene::Scene(void)
{
	envmap = -1;
}


Scene::~Scene(void)
{
}


bool Scene::load(const string& filename)
{
	ifstream fin(filename);
	if( !fin ) {
		cerr << "Failed to load file " << filename << '!' << endl;
		return false;
	}
	while( fin ) {
		string line;
		cout << line << endl;
		getline(fin, line);
		parse(line);
	}
	return true;
}

bool Scene::save(const string& filename)
{
	return false;
}

void Scene::parse(const string& line)
{
	stringstream ss;
	ss<<line;

	string tag;
	ss >> tag;
	cout << tag << endl;

	std::for_each(tag.begin(), tag.end(), ::tolower);
	
	if( tag == "environment" ) {
		string texFile;
		ss >> texFile;
		envmap = loadTexture(texFile.c_str(), texs);
	}
	else if( tag == "camera" ) {
		ss >> cam.pos >> cam.dir >> cam.up >> cam.f >> cam.w >> cam.h;
		cam.dir = cam.dir.normalized();
		cam.up = cam.up.normalized();
	}
	else if( tag == "plane" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		Material mater;
		ss >> mater;
		
		// construct transformation matrix
		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		vec3 n(0, 1, 0), u(1, 0, 0), v(0, 0, 1);
		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createPlane(T, dim.x, dim.y, mrot*n, mrot*u, mrot*v, mater);

		string texFile, normalFile;
		ss >> texFile >> normalFile;
		
		if( texFile != "none" ) {
			sp.hasTexture = true;
			sp.texId = loadTexture(texFile.c_str(), texs);
		}
		if( normalFile != "none" ) {
			sp.hasNormalMap = true;
			sp.normalTexId = loadTexture(normalFile.c_str(), texs);
		}

		shapes.push_back(sp);
	}
	else if( tag == "box" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		Material mater;
		ss >> mater;

		// construct transformation matrix
		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		mat3 M = mrot * mscl;

		vec3 dim = mscl * vec3(1, 1, 1);
		vec3 p, n, u, v;

		// top
		p = vec3(0, 1.0, 0), n = vec3(0, 1, 0), u = vec3(1, 0, 0), v = vec3(0, 0, 1);		
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, mrot*n, mrot*u, mrot*v, mater));
		// bottom
		p = vec3(0, -1.0, 0), n = vec3(0, -1, 0), u = vec3(-1, 0, 0), v = vec3(0, 0, -1);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, mrot*n, mrot*u, mrot*v, mater));
		// left
		p = vec3(-1.0, 0, 0), n = vec3(-1, 0, 0), u = vec3(0, 1, 0), v = vec3(0, 0, 1);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, mrot*n, mrot*u, mrot*v, mater));
		// right
		p = vec3(1.0, 0, 0), n = vec3(1, 0, 0), u = vec3(0, -1, 0), v = vec3(0, 0, -1);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, mrot*n, mrot*u, mrot*v, mater));
		// front
		p = vec3(0, 0, 1.0), n = vec3(0, 0, 1), u = vec3(0, 1, 0), v = vec3(1, 0, 0);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, mrot*n, mrot*u, mrot*v, mater));
		// back
		p = vec3(0, 0, -1.0), n = vec3(0, 0, -1), u = vec3(0, -1, 0), v = vec3(-1, 0, 0);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, mrot*n, mrot*u, mrot*v, mater));
	}
	else if( tag == "sphere" ) {

		vec3 T, S, R;
		ss >> T >> S >> R;
		Material mater;
		ss >> mater;

		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		mat3 M = mrot * mscl;

		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createSphere(T, dim.x, mater);

		string texFile, normalFile;
		ss >> texFile >> normalFile;
		
		if( texFile != "none" ) {
			sp.hasTexture = true;
			sp.texId = loadTexture(texFile.c_str(), texs);
			cout << sp.texId << endl;
		}
		if( normalFile != "none" ) {
			sp.hasNormalMap = true;
			sp.normalTexId = loadTexture(normalFile.c_str(), texs);
		}

		shapes.push_back(sp);
	}
	else if( tag == "ellipsoid") {
		vec3 T, S, R;
		ss >> T >> S >> R;
		Material mater;
		ss >> mater;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		Shape sp = Shape::createEllipsoid(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), mater);

		string texFile, normalFile;
		ss >> texFile >> normalFile;
		
		if( texFile != "none" ) {
			sp.hasTexture = true;
			sp.texId = loadTexture(texFile.c_str(), texs);
			cout << sp.texId << endl;
		}
		if( normalFile != "none" ) {
			sp.hasNormalMap = true;
			sp.normalTexId = loadTexture(normalFile.c_str(), texs);
		}

		shapes.push_back(sp);
	}
	else if( tag == "cylinder" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		Material mater;
		ss >> mater;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		shapes.push_back(Shape::createCylinder(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), mater));
	}
	else if( tag == "cone" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		Material mater;
		ss >> mater;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		shapes.push_back(Shape::createCone(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), mater));
	}
	else if( tag == "hyperboloid" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		Material mater;
		ss >> mater;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		shapes.push_back(Shape::createHyperboloid(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), mater));
	}
	else if( tag == "hyperboloid2" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		Material mater;
		ss >> mater;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		shapes.push_back(Shape::createHyperboloid2(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), mater));
	}
	else if( tag == "mesh" ) {

	}
	else return;
}

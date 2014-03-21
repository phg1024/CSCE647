#include "Scene.h"
#include "utils.h"
#include "extras/tinyobjloader/tiny_obj_loader.h"

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

bool Scene::save(const string& filename){
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

		// test if it is an hdr image
		if( isHDRFile(texFile) ) {
			envmap = loadHDRTexture(texFile, texs);
		}
		else {
			envmap = loadTexture(texFile.c_str(), texs);
		}
	}
	else if( tag == "camera" ) {
		ss >> cam.pos >> cam.dir >> cam.up >> cam.f >> cam.w >> cam.h;
		cam.dir = cam.dir.normalized();
		cam.up = cam.up.normalized();
	}
	else if( tag == "material" ) {
		Material mater;
		ss >> mater.name >> mater;

		if( mater.diffuseTexName != "none" ) {
			mater.diffuseTex = TextureObject::parseType(mater.diffuseTexName);
			if( mater.diffuseTex == TextureObject::Image ){ 
				// load texture from image file
				if( mater.isSolidTex ) 
					mater.diffuseTex += loadTexture(mater.diffuseTexName.c_str(), texs);
				else
					mater.diffuseTex = loadTexture(mater.diffuseTexName.c_str(), texs);
			}
		}
		else mater.diffuseTex = -1;
		if( mater.normalTexName != "none" ) {
			mater.normalTex = TextureObject::parseType(mater.normalTexName);
			if( mater.normalTex == TextureObject::Image ){ 
				mater.normalTex = loadTexture(mater.normalTexName.c_str(), texs);
			}
		}
		else mater.normalTex = -1;

		materials.push_back(mater);
		materialMap[mater.name] = materials.size()-1;
	}
	else if( tag == "plane" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		string matName;
		ss >> matName;
		
		// construct transformation matrix
		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		vec3 n(0, 1, 0), u(1, 0, 0), v(0, 0, 1);
		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createPlane(T, dim.x, dim.y, dim.z, mrot*n, mrot*u, mrot*v, materialMap[matName]);

		shapes.push_back(sp);
	}
	else if( tag == "box" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		string matName;
		ss >> matName;

		// construct transformation matrix
		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		mat3 M = mrot * mscl;

		vec3 dim = mscl * vec3(1, 1, 1);
		vec3 p, n, u, v;

		// top
		p = vec3(0, 1.0, 0), n = vec3(0, 1, 0), u = vec3(1, 0, 0), v = vec3(0, 0, 1);		
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, 1.0, mrot*n, mrot*u, mrot*v, materialMap[matName]));
		// bottom
		p = vec3(0, -1.0, 0), n = vec3(0, -1, 0), u = vec3(-1, 0, 0), v = vec3(0, 0, -1);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, 1.0, mrot*n, mrot*u, mrot*v, materialMap[matName]));
		// left
		p = vec3(-1.0, 0, 0), n = vec3(-1, 0, 0), u = vec3(0, 1, 0), v = vec3(0, 0, 1);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, 1.0, mrot*n, mrot*u, mrot*v, materialMap[matName]));
		// right
		p = vec3(1.0, 0, 0), n = vec3(1, 0, 0), u = vec3(0, -1, 0), v = vec3(0, 0, -1);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, 1.0, mrot*n, mrot*u, mrot*v, materialMap[matName]));
		// front
		p = vec3(0, 0, 1.0), n = vec3(0, 0, 1), u = vec3(0, 1, 0), v = vec3(1, 0, 0);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, 1.0, mrot*n, mrot*u, mrot*v, materialMap[matName]));
		// back
		p = vec3(0, 0, -1.0), n = vec3(0, 0, -1), u = vec3(0, -1, 0), v = vec3(-1, 0, 0);
		shapes.push_back(Shape::createPlane(M*p + T, dim.x, dim.y, 1.0, mrot*n, mrot*u, mrot*v, materialMap[matName]));
	}
	else if( tag == "sphere" ) {

		vec3 T, S, R;
		string matName;
		ss >> T >> S >> R >> matName;
		cout << matName << endl;

		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		mat3 M = mrot * mscl;

		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createSphere(T, dim.x, materialMap[matName]);

		shapes.push_back(sp);
	}
	else if( tag == "ellipsoid") {
		vec3 T, S, R;
		ss >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		Shape sp = Shape::createEllipsoid(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]);

		shapes.push_back(sp);
	}
	else if( tag == "cylinder" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		shapes.push_back(Shape::createCylinder(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]));
	}
	else if( tag == "cone" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		shapes.push_back(Shape::createCone(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]));
	}
	else if( tag == "hyperboloid" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		shapes.push_back(Shape::createHyperboloid(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]));
	}
	else if( tag == "hyperboloid2" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		shapes.push_back(Shape::createHyperboloid2(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]));
	}
	else if( tag == "mesh" ) {
		vec3 T, S, R;
		ss >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		mat3 M = mrot * mscl;

		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createMesh(T, S, mrot, materialMap[matName]);

		string meshFile, texFile, normalFile;
		int solidFlag;
		ss >> meshFile >> texFile >> solidFlag >> normalFile;
		
		// load the mesh and convert it to a texture
		vector<tinyobj::shape_t> objs;
		cout << tinyobj::LoadObj(objs, meshFile.c_str(), "./meshes/") << endl;
		cout << objs.size() << " shapes in total." << endl;
		
		vector<float4> triangles;
		for(int i=0;i<objs.size();i++) {
			const tinyobj::shape_t& shp = objs[i];

			//
		}


		if( texFile != "none" ) {
			sp.material.diffuseTex = TextureObject::parseType(texFile);
			if( sp.material.diffuseTex == TextureObject::Image ){ 
				// load texture from image file
				if( solidFlag ) 
					sp.material.diffuseTex += loadTexture(texFile.c_str(), texs);
				else
					sp.material.diffuseTex = loadTexture(texFile.c_str(), texs);
			}
		}
		else sp.material.diffuseTex = -1;
		if( normalFile != "none" ) {
			sp.material.normalTex = loadTexture(normalFile.c_str(), texs);
		}
		else sp.material.normalTex = -1;

		shapes.push_back(sp);
	}
	else return;
}

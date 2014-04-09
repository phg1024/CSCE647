#include "Scene.h"
#include "utils.h"
#include "extras/tinyobjloader/tiny_obj_loader.h"
#include "extras/aabbtree/aabbtree.h"

Scene::Scene(void)
{
	envmap = -1;	// by default, no environment mapping
	name = "scene"; // default name
	gamma = 1.0;	// default gamma
	ttype = 1;
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
	
	if( tag == "name" ) {
		ss >> name;
	}
	if( tag == "tracingtype" ) {
		ss >> ttype;
	}
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
	else if( tag == "maxiters" ) {
		ss >> maxiters;
	}
	else if( tag == "gamma" ) {
		ss >> gamma;
	}
	else if( tag == "canvas" ) {
		ss >> w >> h;
	}
	else if( tag == "camera" ) {
		ss >> cam.pos >> cam.dir >> cam.up >> cam.f >> cam.fov;
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
		mat3 M = mscl * mrot;

		vec3 n(0, 1, 0), u(1, 0, 0), v(0, 0, 1);
		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createPlane(T, dim.x, dim.y, dim.z, mrot*n, mrot*u, mrot*v, materialMap[matName]);

		sp.bb.minPt = mrot * make_float3(-S.x, -1e-1, -S.z);
		sp.bb.maxPt = mrot * make_float3(S.x, 1e-1, S.z);

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

		S = S * 1.5;
		sp.bb.maxPt = make_float3(T.x + S.x, T.y + S.y, T.z + S.z);
		sp.bb.minPt = make_float3(T.x - S.x, T.y - S.y, T.z - S.z);

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
		string meshFile, matName;
		ss >> meshFile >> matName;

		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		mat3 M = mrot * mscl;

		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createMesh(T, S, mrot, materialMap[matName]);
		
		// load the mesh and convert it to a texture
		vector<tinyobj::shape_t> objs;
		int lastSlashPos = meshFile.find_last_of("/");
		string basePath = meshFile.substr(0, lastSlashPos+1);
		cout << "base path: " << basePath << endl;
		cout << tinyobj::LoadObj(objs, meshFile.c_str(), basePath.c_str()) << endl;
		cout << objs.size() << " shapes in total." << endl;

		// count triangle number
		int ntris = 0;
		for(int i=0,tidx=0;i<objs.size();i++) {
			const tinyobj::shape_t& shp = objs[i];
			const tinyobj::mesh_t& msh = shp.mesh;
			ntris += msh.indices.size() / 3;
		}

		cout << "number of triangles in the mesh: " << ntris << endl;
		
		vector<float4> triangles;
		triangles.reserve(ntris);
		vector<aabbtree::Triangle> tris;		// for building AABB tree
		tris.reserve(ntris);
		vector<float3> normals;
		normals.reserve(ntris);
		vector<float2> texcoords;
		texcoords.reserve(ntris);
		cout << "space reserved for processing the mesh." << endl;

		float3 maxPt = make_float3(-FLT_MAX), minPt = make_float3(FLT_MAX);

		for(int i=0,tidx=0;i<objs.size();i++) {
			const tinyobj::shape_t& shp = objs[i];

			const tinyobj::mesh_t& msh = shp.mesh;
			const tinyobj::material_t& mt = shp.material;

			bool hasNormal = !msh.normals.empty();
			bool hasTexCoords = !msh.texcoords.empty();

			for(int j=0;j<msh.indices.size();j+=3) {
				float3 v0 = make_float3(msh.positions[msh.indices[j]*3], msh.positions[msh.indices[j]*3+1], msh.positions[msh.indices[j]*3+2]);
				float3 v1 = make_float3(msh.positions[msh.indices[j+1]*3], msh.positions[msh.indices[j+1]*3+1], msh.positions[msh.indices[j+1]*3+2]);
				float3 v2 = make_float3(msh.positions[msh.indices[j+2]*3], msh.positions[msh.indices[j+2]*3+1], msh.positions[msh.indices[j+2]*3+2]);

				v0 = M * v0 + T.data;
				v1 = M * v1 + T.data;
				v2 = M * v2 + T.data;

				maxPt = fmaxf(v0, maxPt); minPt = fminf(v0, minPt);
				maxPt = fmaxf(v1, maxPt); minPt = fminf(v1, minPt);
				maxPt = fmaxf(v2, maxPt); minPt = fminf(v2, minPt);
				
				triangles.push_back(make_float4(v0, i));
				triangles.push_back(make_float4(v1, i));
				triangles.push_back(make_float4(v2, i));

				float3 n0 = aabbtree::zero3, n1 = aabbtree::zero3, n2 = aabbtree::zero3;
				if( hasNormal ) {
					n0 = make_float3(msh.normals[msh.indices[j]*3], msh.normals[msh.indices[j]*3+1], msh.normals[msh.indices[j]*3+2]);
					n1 = make_float3(msh.normals[msh.indices[j+1]*3], msh.normals[msh.indices[j+1]*3+1], msh.normals[msh.indices[j+1]*3+2]);
					n2 = make_float3(msh.normals[msh.indices[j+2]*3], msh.normals[msh.indices[j+2]*3+1], msh.normals[msh.indices[j+2]*3+2]);

					n0 = normalize(mrot * n0);
					n1 = normalize(mrot * n1);
					n2 = normalize(mrot * n2);

					normals.push_back(n0);
					normals.push_back(n1);
					normals.push_back(n2);
				}
				//tris.push_back(aabbtree::Triangle(tidx++, v0, v1, v2, n0, n1, n2));
				tris.push_back(aabbtree::Triangle(tidx++, v0, v1, v2));
			}
		}

		sp.bb.minPt = minPt;
		sp.bb.maxPt = maxPt;

		cout << "uploading vertices ..." << endl;
		cout << "copying " << bytes2MB(sizeof(float4)*triangles.size()) << " MB to GPU ..." << endl;
		cudaMalloc(&sp.trimesh.faces, sizeof(float4)*triangles.size());
		cudaMemcpy(sp.trimesh.faces, &triangles[0], sizeof(float4)*triangles.size(), cudaMemcpyHostToDevice);

		if( normals.empty() )
			sp.trimesh.normals = NULL;
		else {
			cout << "uploading normals ..." << endl;
			cout << "copying " << bytes2MB(sizeof(float3)*normals.size()) << " MB to GPU ..." << endl;

			cudaMalloc(&sp.trimesh.normals, sizeof(float3)*normals.size());
			cudaMemcpy(sp.trimesh.normals, &normals[0], sizeof(float3)*normals.size(), cudaMemcpyHostToDevice);
		}
		if( texcoords.empty() ) 
			sp.trimesh.texcoords = NULL;
		else {
			cout << "uploading texture coordinates ..." << endl;
			cout << "copying " << bytes2MB(sizeof(float2)*texcoords.size()) << " MB to GPU ..." << endl;

			cudaMalloc(&sp.trimesh.texcoords, sizeof(float2)*texcoords.size());
			cudaMemcpy(sp.trimesh.texcoords, &texcoords[0], sizeof(float2)*texcoords.size(), cudaMemcpyHostToDevice);
		}

		sp.trimesh.nFaces = triangles.size()/3;

		cout << sp.trimesh.faces << ' '
			 << sp.trimesh.normals << ' '
			 << sp.trimesh.texcoords << ' '
			 << sp.trimesh.nFaces << endl;
		
		// release some memory
		triangles.clear();
		normals.clear();
		texcoords.clear();

		aabbtree::AABBTree tree(tris);
		tree.printNodeStats();
		auto treearray = tree.toArray();

		cout << "uploading aabb tree to device ..." << endl;
		// upload the tree to device
		size_t treesize = sizeof(aabbtree::AABBNode_Serial)*treearray.size();
		cout << "tree size = " << bytes2MB(treesize) << " MB" << endl;
		cudaMalloc(&sp.trimesh.tree, treesize);		
		cudaMemcpy(sp.trimesh.tree, &treearray[0], treesize, cudaMemcpyHostToDevice);
		cout << "done." << endl;

		shapes.push_back(sp);
	}
	else return;
}

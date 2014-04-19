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
	maxbounces = 64;
	interval = 1;
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
	else if( tag == "save_interval" ) {
		ss >> interval;
	}
	else if( tag == "tracingtype" ) {
		ss >> ttype;
	}
	else if( tag == "environment" ) {
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
	else if( tag == "maxbounces" ) {
		ss >> maxbounces;
	}
	else if( tag == "gamma" ) {
		ss >> gamma;
	}
	else if( tag == "canvas" ) {
		ss >> w >> h;
	}
	else if( tag == "camera" ) {
		float fnumber, magratio;
		ss >> cam.pos >> cam.dir >> cam.up >> cam.f >> cam.fov >> fnumber >> cam.magRatio;
		cam.apertureRadius = cam.f / (2.0 * fnumber);
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
		vec3 V, T, S, R;
		ss >> V >> T >> S >> R;
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
		sp.v = V;

		shapes.push_back(sp);
	}
	else if( tag == "sphere" ) {

		vec3 V, T, S, R;
		string matName;
		ss >> V >> T >> S >> R >> matName;
		cout << matName << endl;

		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		mat3 M = mrot * mscl;

		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createSphere(T, dim.x, materialMap[matName]);

		S = S * 1.05;
		sp.bb.maxPt = make_float3(T.x + S.x, T.y + S.y, T.z + S.z);
		sp.bb.minPt = make_float3(T.x - S.x, T.y - S.y, T.z - S.z);

		sp.v = V;

		shapes.push_back(sp);
	}
	else if( tag == "ellipsoid") {
		vec3 V, T, S, R;
		ss >> V >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);


		Shape sp = Shape::createEllipsoid(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]);

		sp.v = V;

		shapes.push_back(sp);
	}
	else if( tag == "cylinder" ) {
		vec3 V, T, S, R;
		ss >> V >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		Shape sp = Shape::createCylinder(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]);
		sp.v = V;
		shapes.push_back(sp);
	}
	else if( tag == "cone" ) {
		vec3 V, T, S, R;
		ss >> V >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		Shape sp = Shape::createCone(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]);
		sp.v = V;
		shapes.push_back(sp);
	}
	else if( tag == "hyperboloid" ) {
		vec3 V, T, S, R;
		ss >> V >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		Shape sp = Shape::createHyperboloid(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]);
		sp.v = V;
		shapes.push_back(sp);
	}
	else if( tag == "hyperboloid2" ) {
		vec3 V, T, S, R;
		ss >> V >> T >> S >> R;
		string matName;
		ss >> matName;

		mat3 mrot = mat3::rotation(R.x, R.y, R.z);

		Shape sp = Shape::createHyperboloid2(T, S, mrot*vec3(1, 0, 0), mrot*vec3(0, 1, 0), mrot*vec3(0, 0, 1), materialMap[matName]);
		sp.v = V;
		shapes.push_back(sp);
	}
	else if( tag == "mesh" ) {
		vec3 V, T, S, R;
		ss >> V >> T >> S >> R;
		string meshFile, matName;
		ss >> meshFile >> matName;

		mat3 mscl = mat3::scaling(S.x, S.y, S.z);
		mat3 mrot = mat3::rotation(R.x, R.y, R.z);
		mat3 M = mrot * mscl;

		vec3 dim = mscl * vec3(1, 1, 1);

		Shape sp = Shape::createMesh(T, S, mrot, materialMap[matName]);
		sp.v = V;
		
		// load the mesh and convert it to a texture
		vector<tinyobj::shape_t> objs;
		int lastSlashPos = meshFile.find_last_of("/");
		string basePath = meshFile.substr(0, lastSlashPos+1);
		cout << "base path: " << basePath << endl;
		cout << tinyobj::LoadObj(objs, meshFile.c_str(), basePath.c_str()) << endl;
		cout << objs.size() << " shapes in total." << endl;

		// count triangle number
		int ntris = 0;
		int nverts = 0;
		for(int i=0,tidx=0;i<objs.size();i++) {
			const tinyobj::shape_t& shp = objs[i];
			const tinyobj::mesh_t& msh = shp.mesh;
			ntris += msh.indices.size() / 3;
			nverts += msh.positions.size() / 3;
		}

		cout << "number of triangles in the mesh: " << ntris << endl;
		cout << "number of vertices in the mesh: " << nverts << endl;
		
		vector<int4> indices;
		indices.reserve(ntris);
		vector<float3> vertices;
		vertices.reserve(nverts);
		vector<aabbtree::Triangle> tris;		// for building AABB tree
		tris.reserve(ntris);
		vector<float3> normals;
		normals.reserve(nverts);
		vector<float2> texcoords;
		texcoords.reserve(nverts);
		cout << "space reserved for processing the mesh." << endl;

		float3 maxPt = make_float3(-FLT_MAX), minPt = make_float3(FLT_MAX);

		int toffset = 0;
		for(int i=0,tidx=0;i<objs.size();i++) {
			const tinyobj::shape_t& shp = objs[i];

			const tinyobj::mesh_t& msh = shp.mesh;
			const tinyobj::material_t& mt = shp.material;

			bool hasNormal = !msh.normals.empty();
			bool hasTexCoords = !msh.texcoords.empty();

			for(int j=0;j<msh.indices.size();j+=3) {
				int4 idx = make_int4(msh.indices[j] + toffset, msh.indices[j+1] + toffset, msh.indices[j+2] + toffset, i);
				if( idx.x >= nverts || idx.y >= nverts || idx.z >= nverts || idx.x < 0 || idx.y < 0 || idx.z < 0 ) cout << "shit" << endl;
				indices.push_back(idx);

				int idx0 = msh.indices[j]*3, idx1 = msh.indices[j+1]*3, idx2 = msh.indices[j+2]*3;
				float3 v0 = make_float3(msh.positions[idx0], msh.positions[idx0+1], msh.positions[idx0+2]);
				float3 v1 = make_float3(msh.positions[idx1], msh.positions[idx1+1], msh.positions[idx1+2]);
				float3 v2 = make_float3(msh.positions[idx2], msh.positions[idx2+1], msh.positions[idx2+2]);

				// transform the vertices to world space
				v0 = M * v0 + T.data;
				v1 = M * v1 + T.data;
				v2 = M * v2 + T.data;

				maxPt = fmaxf(v0, maxPt); minPt = fminf(v0, minPt);
				maxPt = fmaxf(v1, maxPt); minPt = fminf(v1, minPt);
				maxPt = fmaxf(v2, maxPt); minPt = fminf(v2, minPt);
				
				tris.push_back(aabbtree::Triangle(tidx++, v0, v1, v2));
			}

			toffset += msh.positions.size()/3;

			for(int j=0;j<msh.positions.size();j+=3) {
				float3 v = make_float3(msh.positions[j], msh.positions[j+1], msh.positions[j+2]);
				vertices.push_back( M * v + T.data );
			}
			
			if( hasNormal ) {
				for(int j=0;j<msh.normals.size();j+=3) {

					float3 n = make_float3(msh.normals[j], msh.normals[j+1], msh.normals[j+2]);					
					// rotate the normals
					const float THRES = 1e-3;
					if( length(n) > THRES ) n = normalize(mrot * n);
					normals.push_back(n);
				}
			}

			if( hasTexCoords ) {
				for(int j=0;j<msh.texcoords.size();j+=2) {

					float2 t = make_float2(msh.texcoords[j], msh.texcoords[j+1]);					
					//texcoords.push_back(t);
				}
			}
		}

		sp.bb.minPt = minPt;
		sp.bb.maxPt = maxPt;

		cout << "uploading indices ..." << endl;
		cout << "copying " << bytes2MB(sizeof(int4)*indices.size()) << " MB to GPU ..." << endl;
		cudaMalloc(&sp.trimesh.indices, sizeof(int4)*indices.size());
		cudaMemcpy(sp.trimesh.indices, &indices[0], sizeof(int4)*indices.size(), cudaMemcpyHostToDevice);

		cout << "uploading vertices ..." << endl;
		cout << "copying " << bytes2MB(sizeof(float3)*vertices.size()) << " MB to GPU ..." << endl;
		cudaMalloc(&sp.trimesh.verts, sizeof(float3)*vertices.size());
		cudaMemcpy(sp.trimesh.verts, &vertices[0], sizeof(float3)*vertices.size(), cudaMemcpyHostToDevice);

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

		sp.trimesh.nFaces = indices.size();

		cout << sp.trimesh.verts << ' '
			 << sp.trimesh.normals << ' '
			 << sp.trimesh.texcoords << ' '
			 << sp.trimesh.nFaces << endl;
		
		// release some memory
		vertices.clear();
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

#pragma once

#include "element.h"
#include "utils.h"

class Camera {
public:
	Camera(){}
    vec3 pos;
    vec3 up, dir, right;

    float f;
    float w, h;
};

class Light {
public:
	enum LightType {
		POINT = 0,
		DIRECTIONAL,
		SPOT,
		SPHERE,
		AREA
	};

	LightType t;
	float intensity;
	vec3 pos;
	vec3 dir;
	float radius;

	float spotExponent;
	float spotCutOff;

	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	vec3 attenuation;

	__device__ __host__ Light(){}
	__device__ __host__ Light(const Light& lt):
		t(lt.t), intensity(lt.intensity), ambient(lt.ambient), diffuse(lt.diffuse), specular(lt.specular), 
		pos(lt.pos), dir(lt.dir), radius(lt.radius), spotExponent(lt.spotExponent), spotCutOff(lt.spotCutOff), attenuation(lt.attenuation){}
	__device__ __host__ ~Light(){}

	// point light
	__device__ __host__ Light(LightType t, float intensity, vec3 ambient, vec3 diffuse, vec3 specular, vec3 position):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), pos(position){}

	// sphere light
	__device__ __host__ Light(LightType t, float intensity, vec3 ambient, vec3 diffuse, vec3 specular, vec3 position, float r = 1.0):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), pos(position), radius(r){}

	// directional light
	__device__ __host__ Light(LightType t, float intensity, vec3 ambient, vec3 diffuse, vec3 specular, vec3 position, vec3 direction):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), pos(position), dir(direction.normalized()){}

	// spot light
	__device__ __host__ Light(LightType t, float intensity, vec3 ambient, vec3 diffuse, vec3 specular, vec3 position, vec3 direction, float expo, float cutoff):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), pos(position), dir(direction), spotExponent(expo), spotCutOff(cutoff){}
};

struct d_Light {
	__device__ void init(const d_Light& m) {
		t = m.t;
		intensity = m.intensity;
		pos = m.pos;
		dir = m.dir;
		radius = m.radius;
		spotExponent = m.spotExponent;
		spotCutOff = m.spotCutOff;

		ambient = m.ambient;
		diffuse = m.diffuse;
		specular = m.specular;
		attenuation = m.attenuation;
	}

	__device__ void init(const Light& m) {
		t = m.t;
		intensity = m.intensity;
		pos = m.pos.data;
		dir = m.dir.data;
		radius = m.radius;
		spotExponent = m.spotExponent;
		spotCutOff = m.spotCutOff;

		ambient = m.ambient.data;
		diffuse = m.diffuse.data;
		specular = m.specular.data;
		attenuation = m.attenuation.data;
	}

	Light::LightType t;
	float intensity;
	float3 pos;
	float3 dir;
	float radius;

	float spotExponent;
	float spotCutOff;

	float3 ambient;
	float3 diffuse;
	float3 specular;

	float3 attenuation;
};

class Material {
public:
	enum MaterialType {
		Emissive,
		Diffuse,
		Specular,
		Refractive,
		DiffuseScatter,
		Glossy
	};
	static Material makeDiffuseScatter(vec3 diffuse, float r) {
		Material m;
		m.diffuse = diffuse;
		m.t = DiffuseScatter;
		m.Kr = r;
		return m;
	}
	static Material makeGlossy(vec3 v, float r) {
		Material m;
		m.diffuse = v;
		m.t = Glossy;
		m.Kr = r;
		return m;
	}
	static Material makeDiffuse(vec3 v) {
		Material m;
		m.diffuse = v;
		m.t = Diffuse;
		return m;
	}
	static Material makeSpecular(vec3 v) {
		Material m;
		m.diffuse = v;
		m.t = Specular;
		return m;
	}
	static Material makeEmissive(vec3 v) {
		Material m;
		m.emission = v;
		m.t = Emissive;
		return m;
	}
	static Material makeRefractive(vec3 v) {
		Material m;
		m.diffuse = v;
		m.eta = 1.5;
		m.t = Refractive;
		return m;
	}
	__device__ __host__ Material():diffuse(vec3(1, 1, 1)), specular(vec3(1, 1, 1)), ambient(vec3(0.05, 0.05, 0.05)), shininess(50.0),
	kcool(vec3(0, 0, 0.75)), kwarm(vec3(0.75, 0.75, 0)), alpha(0.15f), beta(0.25f), Ks(1.0), Kr(0.0), Kf(0.0), eta(1.0){}
	__device__ __host__ Material(
		vec3 diffuse, vec3 specular, vec3 ambient, float shininess,
		vec3 kcool, vec3 kwarm, float alpha = 0.15f, float beta = 0.25f,
		float ks = 1.0, float kr = 0.0, float kf = 0.0, float eta = 1.1,
		MaterialType t = Diffuse
		):
		t(t),
		emission(vec3(0, 0, 0)), ambient(ambient), diffuse(diffuse), specular(specular), shininess(shininess), eta(eta),
		kcool(kcool), kwarm(kwarm), alpha(alpha), beta(beta)
	{
		Ks = ks; Kr = kr; Kf = kf;
		}
	__device__ __host__ Material(
			vec3 diffuse, vec3 specular, vec3 ambient, vec3 emission, float shininess,
			vec3 kcool, vec3 kwarm, float alpha = 0.15f, float beta = 0.25f,
			float ks = 1.0, float kr = 0.0, float kf = 0.0, float eta = 1.1
			):
		t(Emissive),
		emission(emission), ambient(ambient), diffuse(diffuse), specular(specular), shininess(shininess), eta(eta),
			kcool(kcool), kwarm(kwarm), alpha(alpha), beta(beta)
		{
			Ks = ks; Kr = kr; Kf = kf;
		}
	__device__ __host__ Material(const Material& m):
		t(m.t), emission(m.emission), ambient(m.ambient), diffuse(m.diffuse), specular(m.specular), shininess(m.shininess), eta(m.eta),
		Ks(m.Ks), Kr(m.Kr), Kf(m.Kf), kcool(m.kcool), kwarm(m.kwarm), alpha(m.alpha), beta(m.beta)
	{}
	__device__ __host__ Material& operator=(const Material& m) {
		t = m.t; emission = m.emission; ambient = m.ambient; diffuse = m.diffuse; specular = m.specular;
		shininess = m.shininess; eta = m.eta; Ks = m.Ks; Kr = m.Kr; Kf = m.Kf; 
		kcool = m.kcool; kwarm = m.kwarm; alpha = m.alpha; beta = m.beta;

		return (*this);
	}

	__device__ __host__ ~Material(){}

	static __host__ MaterialType string2type(const string& s) {
		string tag = s;
		std::for_each(tag.begin(), tag.end(), ::tolower);

		if( tag == "emissive" ) {
			return Emissive;
		}
		else if( tag == "diffuse" ) {
			return Diffuse;
		}
		else if( tag == "specular" ) {
			return Specular;
		}
		else if( tag == "refractive" ) {
			return Refractive;
		}
		else if( tag == "diffusescatter" ) {
			return DiffuseScatter;
		}
		else if( tag == "glossy" ) {
			return Glossy;
		}
		else return Diffuse;
	}

	__host__ friend istream& operator>>(istream& is, Material& mater);

	MaterialType t;
	vec3 emission;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	float Ks, Kr, Kf;
	float shininess;
	float eta;

	vec3 kcool, kwarm;
	float alpha, beta;
};

__host__ __inline__ istream& operator>>(istream& is, Material& mater) {
	string tag;
	is >> tag >> mater.emission >> mater.ambient >> mater.diffuse >> mater.specular >> mater.Ks >> mater.Kr
		>> mater.Kf >> mater.shininess >> mater.eta >> mater.kcool >> mater.kwarm >> mater.alpha >> mater.beta;

	mater.t = Material::string2type(tag);
	cout << tag << endl;
	return is;
}

struct d_Material {
	__device__ void init(const Material& m) {
		t = m.t;
		emission = m.emission.data;
		ambient = m.ambient.data;
		diffuse = m.diffuse.data;
		specular = m.specular.data;

		Ks = m.Ks;
		Kr = m.Kr;
		Kf = m.Kf;
		shininess = m.shininess;
		eta = m.eta;
		kcool = m.kcool.data;
		kwarm = m.kwarm.data;

		alpha = m.alpha;
		beta = m.beta;
	}

	Material::MaterialType t;
	float3 emission;
	float3 ambient;
	float3 diffuse;
	float3 specular;

	float Ks, Kr, Kf;
	float shininess;
	float eta;

	float3 kcool, kwarm;
	float alpha, beta;
};

class Shape {
public:
	enum ShapeType {
		PLANE = 0,
		ELLIPSOID = 1,
		CYLINDER,
		CONE,
		HYPERBOLOID,
		HYPERBOLOID2,
		TRIANGLE_MESH
	};
    __device__ __host__ Shape(){}
	__device__ __host__ Shape(ShapeType t, vec3 p, float r0, float r1, float r2,
		  vec3 a0, vec3 a1, vec3 a2,
		  Material mater):
	t(t), p(p), material(mater), hasTexture(false), texId(-1), hasNormalMap(false), normalTexId(-1) {
		a0 = a0.normalized(); a1 = a1.normalized(); a2 = a2.normalized();
		axis[0] = a0; axis[1] = a1; axis[2] = a2;
		radius[0] = r0; radius[1] = r1; radius[2] = r2;

		vec3 ratio0 = a0/r0;
		vec3 ratio1 = a1/r1;
		vec3 ratio2 = a2/r2;
		
		if( t == ELLIPSOID ) m = outerProduct(ratio0, ratio0) + outerProduct(ratio1, ratio1) + outerProduct(ratio2, ratio2);
		else if( t == CYLINDER ) m = outerProduct(ratio1, ratio1) + outerProduct(ratio2, ratio2);
		else if(t == HYPERBOLOID  || t == CONE) {
			m = -outerProduct(ratio0, ratio0) + outerProduct(ratio1, ratio1) + outerProduct(ratio2, ratio2);		
		}
		else if( t == HYPERBOLOID2 ) {
			m = outerProduct(ratio0, ratio0) - outerProduct(ratio1, ratio1) - outerProduct(ratio2, ratio2);		
		}
	}

	__device__ __host__ Shape(const Shape& s):t(s.t), p(s.p), m(s.m), 
		material(s.material), hasTexture(s.hasTexture), texId(s.texId), hasNormalMap(s.hasNormalMap),
	normalTexId(s.normalTexId){
		for(int i=0;i<3;i++) {
			axis[i] = s.axis[i];
			radius[i] = s.radius[i];
		}
	}
	__device__ __host__ ~Shape(){}

	static Shape createSphere(vec3 center, float radius, Material mater){
		return Shape(ELLIPSOID, center, radius, radius, radius, vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1),
			mater);
	}

	static Shape createEllipsoid(vec3 center, vec3 rad, vec3 a0, vec3 a1, vec3 a2, Material mater) {
		return Shape(ELLIPSOID, center, rad.x, rad.y, rad.z, a0, a1, a2, mater);
	}

	static Shape createPlane(vec3 center, float w, float h, vec3 normal, vec3 u, vec3 v, Material mater) {
		return Shape(PLANE, center, w, h, 0.0, normal, u, v, mater);
	}

	static Shape createCylinder(vec3 center, vec3 rad, vec3 a0, vec3 a1, vec3 a2, Material mater){
		return Shape(CYLINDER, center, rad.x, rad.y, rad.z, a0, a1, a2, mater);
	}

	static Shape createCone(vec3 center, vec3 rad, vec3 a0, vec3 a1, vec3 a2, Material mater) {
		return Shape(CONE, center, rad.x, rad.y, rad.z, a0, a1, a2, mater);
	}

	static Shape createHyperboloid(vec3 center, vec3 rad, vec3 a0, vec3 a1, vec3 a2, Material mater) {
		return Shape(HYPERBOLOID, center, rad.x, rad.y, rad.z, a0, a1, a2, mater);
	}

	static Shape createHyperboloid2(vec3 center, vec3 rad, vec3 a0, vec3 a1, vec3 a2, Material mater) {
		return Shape(HYPERBOLOID2, center, rad.x, rad.y, rad.z, a0, a1, a2, mater);
	}

	ShapeType t;

	// geometry
	vec3 p;
	vec3 axis[3];
	float radius[3];
	mat3 m;

	Material material;

	bool hasTexture;
	int texId;
	bool hasNormalMap;
	int normalTexId;
};

struct d_Shape {
	Shape::ShapeType t;

	__device__ void init(const Shape& s) {
		t = s.t;
		p = s.p.data;
		axis[0] = s.axis[0].data;
		axis[1] = s.axis[1].data;
		axis[2] = s.axis[2].data;
		
		radius[0] = s.radius[0];
		radius[1] = s.radius[1];
		radius[2] = s.radius[2];

		for(int i=0;i<9;i++) m[i] = s.m(i);
		material.init(s.material);

		hasTexture = s.hasTexture;
		texId = s.texId;
		hasNormalMap = s.hasNormalMap;
		normalTexId = s.normalTexId;

		constant = (t==Shape::HYPERBOLOID2)?-1.0:1.0;
		constant2 = (t==Shape::CONE)?0.0:1.0;
	}

	__device__ float3 randomPointOnSurface(int2 res, float time, int x, int y) const {
		switch( t ) {
		case Shape::ELLIPSOID: 
			{
				float2 uv = generateRandomNumberFromThread2(res, time, x, y);
				float3 sp = calculateRandomDirectionInSphere(uv.x, uv.y);
				return (sp.x * axis[0] * radius[0] + sp.y * axis[1] * radius[1] + sp.z * axis[2] * radius[2]) + p;
			}
		case Shape::PLANE:
			{
				float2 uv = generateRandomOffsetFromThread2(res, time, x, y);
				return uv.x * axis[1] + uv.y * axis[2] + p + 1e-3 * axis[0];
			}
		default:
			return p;
		}
	}

	// geometry
	float3 p;
	float3 axis[3];
	float radius[3];
	float m[9];
	float constant;
	float constant2;

	d_Material material;

	bool hasTexture;
	int texId;
	bool hasNormalMap;
	int normalTexId;
};

struct Ray {
	float3 origin;
	float3 dir;
};

struct Hit {
	float t;
	float3 p;		// hit point
	float3 n;		// normal vector
	float2 tex;		// texture coordinates
	int objIdx;		// hit object index
};
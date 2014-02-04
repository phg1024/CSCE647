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
		SPOT
	};

	LightType t;
	float intensity;
	vec3 pos;
	vec3 dir;

	float spotExponent;
	float spotCutOff;

	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	vec3 attenuation;

	__device__ __host__ Light(){}
	__device__ __host__ Light(const Light& lt):
		t(lt.t), intensity(lt.intensity), ambient(lt.ambient), diffuse(lt.diffuse), specular(lt.specular), 
		pos(lt.pos), dir(lt.dir), spotExponent(lt.spotExponent), spotCutOff(lt.spotCutOff), attenuation(lt.attenuation){}
	__device__ __host__ ~Light(){}

	// point light
	__device__ __host__ Light(LightType t, float intensity, vec3 ambient, vec3 diffuse, vec3 specular, vec3 position):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), pos(position){}

	// directional light
	__device__ __host__ Light(LightType t, float intensity, vec3 ambient, vec3 diffuse, vec3 specular, vec3 position, vec3 direction):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), pos(position), dir(direction){}

	// spot light
	__device__ __host__ Light(LightType t, float intensity, vec3 ambient, vec3 diffuse, vec3 specular, vec3 position, vec3 direction, float expo, float cutoff):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), pos(position), dir(direction), spotExponent(expo), spotCutOff(cutoff){}
};

struct d_Light {
	__device__ void init(const Light& m) {
		t = m.t;
		intensity = m.intensity;
		pos = m.pos.data;
		dir = m.dir.data;
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

	float spotExponent;
	float spotCutOff;

	float3 ambient;
	float3 diffuse;
	float3 specular;

	float3 attenuation;
};

class Material {
public:
	__device__ __host__ Material(){}
	__device__ __host__ Material(
		vec3 diffuse, vec3 specular, vec3 ambient, float shininess,
		vec3 kcool, vec3 kwarm, float alpha = 0.15f, float beta = 0.25f,
		float ks = 1.0, float kr = 0.0, float kf = 0.0
		):
		ambient(ambient), diffuse(diffuse), specular(specular), shininess(shininess),
		kcool(kcool), kwarm(kwarm), alpha(alpha), beta(beta)
	{
		Ks = ks; Kr = kr; Kf = kf;
		}
	__device__ __host__ Material(const Material& m):
		emission(m.emission), ambient(m.ambient), diffuse(m.diffuse), specular(m.specular), shininess(m.shininess),
		Ks(m.Ks), Kr(m.Kr), Kf(m.Kf), kcool(m.kcool), kwarm(m.kwarm), alpha(m.alpha), beta(m.beta)
	{}
	__device__ __host__ Material& operator=(const Material& m) {
		emission = m.emission; ambient = m.ambient; diffuse = m.diffuse; specular = m.specular;
		shininess = m.shininess; Ks = m.Ks; Kr = m.Kr; Kf = m.Kf; kcool = m.kcool; kwarm = m.kwarm; alpha = m.alpha; beta = m.beta;

		return (*this);
	}

	__device__ __host__ ~Material(){}

	vec3 emission;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	float Ks, Kr, Kf;
	float shininess;

	vec3 kcool, kwarm;
	float alpha, beta;
};

struct d_Material {
	__device__ void init(const Material& m) {
		emission = m.emission.data;
		ambient = m.ambient.data;
		diffuse = m.diffuse.data;
		specular = m.specular.data;

		Ks = m.Ks;
		Kr = m.Kr;
		Kf = m.Kf;
		shininess = m.shininess;
		kcool = m.kcool.data;
		kwarm = m.kwarm.data;

		alpha = m.alpha;
		beta = m.beta;
	}

	float3 emission;
	float3 ambient;
	float3 diffuse;
	float3 specular;

	float Ks, Kr, Kf;
	float shininess;

	float3 kcool, kwarm;
	float alpha, beta;
};

class Shape {
public:
	enum ShapeType {
		SPHERE = 0,
		PLANE = 1,
		ELLIPSOID = 2,
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
		else if(t == HYPERBOLOID) {
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
	}

	// geometry
	float3 p;
	float3 axis[3];
	float radius[3];
	float m[9];

	d_Material material;

	bool hasTexture;
	int texId;
	bool hasNormalMap;
	int normalTexId;
};

struct Ray {
	float3 origin;
	float3 dir;
	int level;
};

struct Hit {
	float3 color;
	float t;
};
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

class Material {
public:
	__device__ __host__ Material(){}
	__device__ __host__ Material(
		vec3 diffuse, vec3 specular, vec3 ambient, float shininess,
		vec3 kcool, vec3 kwarm, float alpha = 0.15f, float beta = 0.25f
		):
		ambient(ambient), diffuse(diffuse), specular(specular), shininess(shininess),
		kcool(kcool), kwarm(kwarm), alpha(alpha), beta(beta)
	{}
	__device__ __host__ Material(const Material& m):
		emission(m.emission), ambient(m.ambient), diffuse(m.diffuse), specular(m.specular), shininess(m.shininess),
		kcool(m.kcool), kwarm(m.kwarm), alpha(m.alpha), beta(m.beta)
	{}
	__device__ __host__ Material& operator=(const Material& m) {
		emission = m.emission; ambient = m.ambient; diffuse = m.diffuse; specular = m.specular;
		shininess = m.shininess; kcool = m.kcool; kwarm = m.kwarm; alpha = m.alpha; beta = m.beta;

		return (*this);
	}

	__device__ __host__ ~Material(){}

	vec3 emission;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	float shininess;

	vec3 kcool, kwarm;
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
		TRIANGLE_MESH
	};
    __device__ __host__ Shape(){}
	__device__ __host__ Shape(ShapeType t, vec3 p, float r0, float r1, float r2,
		  vec3 a0, vec3 a1, vec3 a2,
		  Material mater):
	t(t), p(p), material(mater), hasTexture(false), texId(-1), hasNormalMap(false), normalTexId(-1) {
		axis[0] = a0; axis[1] = a1; axis[2] = a2;
		radius[0] = r0; radius[1] = r1; radius[2] = r2;

		vec3 ratio0 = a0/r0;
		vec3 ratio1 = a1/r1;
		vec3 ratio2 = a2/r2;
		m = outerProduct(ratio0, ratio0) + outerProduct(ratio1, ratio1) + outerProduct(ratio2, ratio2);
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

struct Hit {
	vec3 color;
	float t;
};

struct Ray {
	vec3 origin;
	vec3 dir;
};
#pragma once

#include "common.h"
#include "element.h"
#include "Utils/stringutils.h"
#include <string>
using namespace std;

struct Light {
	enum LightType {
		POINT = 0,
		DIRECTIONAL,
		SPOT
	};

	Light(){}

	// point light
	Light(LightType t, float intensity, float3 ambient, float3 diffuse, float3 specular, float3 position):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), position(position){}

	// directional light
	Light(LightType t, float intensity, float3 ambient, float3 diffuse, float3 specular, float3 position, float3 direction):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), position(position), direction(direction){}

	// spot light
	Light(LightType t, float intensity, float3 ambient, float3 diffuse, float3 specular, float3 position, float3 direction, float expo, float cutoff):
		t(t), intensity(intensity), ambient(ambient), diffuse(diffuse), specular(specular), position(position), direction(direction), spotExponent(expo), spotCutOff(cutoff){}

    string toString(const string& var, int idx) const;

	void uploadToShader(QGLShaderProgram* program, 
						const string& var);
	void uploadToShader(QGLShaderProgram* program,
						const string& var, int idx);

	LightType t;
	float intensity;
	float3 position;
	float3 direction;

	float spotExponent;		// spot light exponent
	float spotCutOff;		// spot light cutoff angle

	float3 ambient;
	float3 diffuse;
	float3 specular;

	float3 attenuation;
};

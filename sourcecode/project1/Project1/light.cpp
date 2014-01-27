#include "light.h"

void Light::uploadToShader(QGLShaderProgram* program, const string& var)
{	
	string str;	
	str = var + ".type";
	program->setUniformValue(str.c_str(), t);
	str = var + ".intensity";
	program->setUniformValue(str.c_str(), 0.75f);
	str = var + ".ambient";
	program->setUniformValue(str.c_str(), ambient.toQVector());
	str = var + ".diffuse";
	program->setUniformValue(str.c_str(), diffuse.toQVector());
	str = var + ".specular";
	program->setUniformValue(str.c_str(), specular.toQVector());

	str = var + ".pos";
	program->setUniformValue(str.c_str(), position.toQVector());
	str = var + ".dir";
	program->setUniformValue(str.c_str(), direction.toQVector());
}

void Light::uploadToShader(QGLShaderProgram* program, const string& var, int idx)
{
	string str;	
	str = var + "[" + PhGUtils::toString(idx) + "].type";
	program->setUniformValue(str.c_str(), t);
	str = var + "[" + PhGUtils::toString(idx) + "].intensity";
	program->setUniformValue(str.c_str(), intensity);
	str = var + "[" + PhGUtils::toString(idx) + "].ambient";
	program->setUniformValue(str.c_str(), ambient.toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].diffuse";
	program->setUniformValue(str.c_str(), diffuse.toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].specular";
	program->setUniformValue(str.c_str(), specular.toQVector());

	str = var + "[" + PhGUtils::toString(idx) + "].pos";
	program->setUniformValue(str.c_str(), position.toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].dir";
	program->setUniformValue(str.c_str(), direction.toQVector());
}

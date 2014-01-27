#include "shape.h"

void Shape::uploadToShader(QGLShaderProgram *program, const string& var)
{
	string str;
	str = var + ".type";
	program->setUniformValue(str.c_str(), t);

	// geometric info
	str = var + ".p";
	program->setUniformValue(str.c_str(), p.toQVector());
	str = var + ".axis[0]";
	program->setUniformValue(str.c_str(), axis[0].toQVector());
	str = var + ".axis[1]";
	program->setUniformValue(str.c_str(), axis[1].toQVector());
	str = var + ".axis[2]";
	program->setUniformValue(str.c_str(), axis[2].toQVector());
	str = var + ".radius[0]";
	program->setUniformValue(str.c_str(), radius[0]);
	str = var + ".radius[1]";
	program->setUniformValue(str.c_str(), radius[1]);
	str = var + ".radius[2]";
	program->setUniformValue(str.c_str(), radius[2]);
	str = var + ".m";
	program->setUniformValue(str.c_str(), m.toQMatrix());


	// material info
	str = var + ".diffuse";
	program->setUniformValue(str.c_str(), material.diffuse.toQVector());
	str = var + ".specular";
	program->setUniformValue(str.c_str(), material.specular.toQVector());
	str = var + ".ambient";
	program->setUniformValue(str.c_str(), material.ambient.toQVector());
	str = var + ".shininess";
	program->setUniformValue(str.c_str(), material.shininess);
	str = var + ".kcool";
	program->setUniformValue(str.c_str(), material.kcool.toQVector());
	str = var + ".kwarm";
	program->setUniformValue(str.c_str(), material.kwarm.toQVector());

	// texture
	str = var + ".hasTexture";
	program->setUniformValue(str.c_str(), hasTexture);
	str = var + ".tex";
	program->setUniformValue(str.c_str(), texId);

	// normal map
	str = var + ".hasNormalMap";
	program->setUniformValue(str.c_str(), hasNormalMap);
	str = var + ".nTex";
	program->setUniformValue(str.c_str(), normalTexId);
}

void Shape::uploadToShader(QGLShaderProgram *program, const string& var, int idx)
{
	string str;
	str = var + "[" + PhGUtils::toString(idx) + "].type";
	program->setUniformValue(str.c_str(), t);

	// geometric info
	str = var + "[" + PhGUtils::toString(idx) + "].p";
	program->setUniformValue(str.c_str(), p.toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].axis[0]";
	program->setUniformValue(str.c_str(), axis[0].toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].axis[1]";
	program->setUniformValue(str.c_str(), axis[1].toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].axis[2]";
	program->setUniformValue(str.c_str(), axis[2].toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].radius[0]";
	program->setUniformValue(str.c_str(), radius[0]);
	str = var + "[" + PhGUtils::toString(idx) + "].radius[1]";
	program->setUniformValue(str.c_str(), radius[1]);
	str = var + "[" + PhGUtils::toString(idx) + "].radius[2]";
	program->setUniformValue(str.c_str(), radius[2]);
	str = var + "[" + PhGUtils::toString(idx) + "].m";
	program->setUniformValue(str.c_str(), m.toQMatrix());

	// material info
	str = var + "[" + PhGUtils::toString(idx) + "].diffuse";
	program->setUniformValue(str.c_str(), material.diffuse.toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].specular";
	program->setUniformValue(str.c_str(), material.specular.toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].ambient";
	program->setUniformValue(str.c_str(), material.ambient.toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].shininess";
	program->setUniformValue(str.c_str(), material.shininess);
	str = var + "[" + PhGUtils::toString(idx) + "].kcool";
	program->setUniformValue(str.c_str(), material.kcool.toQVector());
	str = var + "[" + PhGUtils::toString(idx) + "].kwarm";
	program->setUniformValue(str.c_str(), material.kwarm.toQVector());

	// texture
	str = var + "[" + PhGUtils::toString(idx) + "].hasTexture";
	program->setUniformValue(str.c_str(), hasTexture);
	str = var + "[" + PhGUtils::toString(idx) + "].tex";
	program->setUniformValue(str.c_str(), texId);

	// normal map
	str = var + "[" + PhGUtils::toString(idx) + "].hasNormalMap";
	program->setUniformValue(str.c_str(), hasNormalMap);
	str = var + "[" + PhGUtils::toString(idx) + "].nTex";
	program->setUniformValue(str.c_str(), normalTexId);
}

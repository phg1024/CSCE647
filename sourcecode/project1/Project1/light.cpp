#include "light.h"

string Light::toString(const string &var, int idx) const
{
    string str = "Light " + var + PhGUtils::toString(idx) + " = Light(";
    str += PhGUtils::toString(t) + ",\n";

    str += PhGUtils::toString(intensity) + ",\n";

    str += "vec3" + ambient.toString() + ",\n";
    str += "vec3" + diffuse.toString() + ",\n";
    str += "vec3" + specular.toString() + ",\n";

    str += "vec3" + position.toString() + ",\n";
    str += "vec3" + direction.toString() + ",\n";

    str += PhGUtils::toString(spotExponent) + ",\n";
    str += PhGUtils::toString(spotCutOff) + ",\n";
    str += PhGUtils::toString(cos(spotCutOff)) + ",\n";

    str += "vec3" + attenuation.toString();

    str += ");\n";
    return str;
}

void Light::uploadToShader(QGLShaderProgram* program, const string& var)
{
    vector<QVector3D> vecs;

    vecs.push_back(QVector3D(t, 0, 0));
    vecs.push_back(QVector3D(intensity, 0, 0));
    vecs.push_back(ambient.toQVector());
    vecs.push_back(diffuse.toQVector());
    vecs.push_back(specular.toQVector());

    vecs.push_back(position.toQVector());
    vecs.push_back(direction.toQVector());

    vecs.push_back(QVector3D(spotExponent, spotCutOff, cos(spotCutOff)));
    vecs.push_back(attenuation.toQVector());

    program->setUniformValueArray(var.c_str(), &(vecs[0]), vecs.size());
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

#include "light.h"

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

    string str = var + "[" + PhGUtils::toString(idx) + "]";
    program->setUniformValueArray(str.c_str(), &(vecs[0]), vecs.size());
}

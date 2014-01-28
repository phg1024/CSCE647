#include "shape.h"

void Shape::uploadToShader(QGLShaderProgram *program, const string& var)
{
    vector<QVector3D> vecs;

    vecs.push_back(QVector3D(t, 0, 0));
    vecs.push_back(p.toQVector());
    vecs.push_back(axis[0].toQVector());
    vecs.push_back(axis[1].toQVector());
    vecs.push_back(axis[2].toQVector());
    vecs.push_back(QVector3D(radius[0], radius[1], radius[2]));
    vecs.push_back(QVector3D(angle, height, 0));
    vecs.push_back(QVector3D(m(0, 0), m(1, 0), m(2, 0)));
    vecs.push_back(QVector3D(m(0, 1), m(1, 1), m(2, 1)));
    vecs.push_back(QVector3D(m(0, 2), m(1, 2), m(2, 2)));

    vecs.push_back(material.emission.toQVector());
    vecs.push_back(material.ambient.toQVector());
    vecs.push_back(material.diffuse.toQVector());
    vecs.push_back(material.specular.toQVector());

    vecs.push_back(material.kcool.toQVector());
    vecs.push_back(material.kwarm.toQVector());

    vecs.push_back(QVector3D(material.shininess, material.alpha, material.beta));
    vecs.push_back(QVector3D(hasTexture?1.0:0.0, texId, 0));
    vecs.push_back(QVector3D(hasNormalMap?1.0:0.0, normalTexId, 0));

    program->setUniformValueArray(var.c_str(), &(vecs[0]), vecs.size());
}

void Shape::uploadToShader(QGLShaderProgram *program, const string& var, int idx)
{
    vector<QVector3D> vecs;

    vecs.push_back(QVector3D(t, 0, 0));
    vecs.push_back(p.toQVector());
    vecs.push_back(axis[0].toQVector());
    vecs.push_back(axis[1].toQVector());
    vecs.push_back(axis[2].toQVector());
    vecs.push_back(QVector3D(radius[0], radius[1], radius[2]));
    vecs.push_back(QVector3D(angle, height, 0));
    vecs.push_back(QVector3D(m(0, 0), m(1, 0), m(2, 0)));
    vecs.push_back(QVector3D(m(0, 1), m(1, 1), m(2, 1)));
    vecs.push_back(QVector3D(m(0, 2), m(1, 2), m(2, 2)));

    vecs.push_back(material.emission.toQVector());
    vecs.push_back(material.ambient.toQVector());
    vecs.push_back(material.diffuse.toQVector());
    vecs.push_back(material.specular.toQVector());

    vecs.push_back(material.kcool.toQVector());
    vecs.push_back(material.kwarm.toQVector());

    vecs.push_back(QVector3D(material.shininess, material.alpha, material.beta));
    vecs.push_back(QVector3D(hasTexture?1.0:0.0, texId, 0));
    vecs.push_back(QVector3D(hasNormalMap?1.0:0.0, normalTexId, 0));

    for(int i=0;i<vecs.size();i++) {
        string str = var + "[" + PhGUtils::toString(idx+i) + "]";
        //cout << str << " = " << vecs[i].x() << ", " << vecs[i].y() << ", " << vecs[i].z() << endl;
        program->setUniformValue(str.c_str(), vecs[i]);
    }

    // textures
}

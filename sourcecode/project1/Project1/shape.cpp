#include "shape.h"

string Shape::toString(const string& var, int idx) const
{
    string prefix = var + PhGUtils::toString(idx);
    string str = "Shape " + prefix + " = Shape(";
    str += PhGUtils::toString(t) + ",\n";

    // geometric info
    str += "vec3" + p.toString() + ",\n";
    str += "vec3" + axis[0].toString() + ",\n";
    str += "vec3" + axis[1].toString() + ",\n";
    str += "vec3" + axis[2].toString() + ",\n";

    str += "vec3" + float3(radius[0], radius[1], radius[2]).toString() + ",\n";

    str += PhGUtils::toString(angle) + ",\n";
    str += PhGUtils::toString(height) + ",\n";

    str += "mat3" + m.transposed().toString() + ",\n";

    str += "vec3" + material.emission.toString() + ",\n";
    str += "vec3" + material.ambient.toString() + ",\n";
    str += "vec3" + material.diffuse.toString() + ",\n";
    str += "vec3" + material.specular.toString() + ",\n";

    str += "vec3" + material.kcool.toString() + ",\n";
    str += "vec3" + material.kwarm.toString() + ",\n";

    str += PhGUtils::toString(material.shininess) + ",\n";
    str += PhGUtils::toString(material.alpha) + ",\n";
    str += PhGUtils::toString(material.beta) + ",\n";

    str += PhGUtils::toString(hasTexture) + ",\n";
    str += PhGUtils::toString(texId) + ",\n";

    str += PhGUtils::toString(hasNormalMap) + ",\n";
    str += PhGUtils::toString(normalTexId);

    str += ");\n";
    return str;
}

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

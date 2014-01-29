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
    string str;
    str = var + "[" + PhGUtils::toString(idx) + "].type";
    program->setUniformValue(str.c_str(), t);

    // geometric info
    str = var + "[" + PhGUtils::toString(idx) + "].p";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << p << endl;
    program->setUniformValue(str.c_str(), p.toQVector());

    str = var + "[" + PhGUtils::toString(idx) + "].axis0";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << axis[0] << endl;
    program->setUniformValue(str.c_str(), axis[0].toQVector());

    str = var + "[" + PhGUtils::toString(idx) + "].axis1";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << axis[1] << endl;
    program->setUniformValue(str.c_str(), axis[1].toQVector());

    str = var + "[" + PhGUtils::toString(idx) + "].axis2";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << axis[2] << endl;
    program->setUniformValue(str.c_str(), axis[2].toQVector());

    str = var + "[" + PhGUtils::toString(idx) + "].radius";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << radius[0] << endl;
    program->setUniformValue(str.c_str(), QVector3D(radius[0], radius[1], radius[2]));

    str = var + "[" + PhGUtils::toString(idx) + "].m";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << m << endl;
    program->setUniformValue(str.c_str(), m.toQMatrix());

    // material info
    str = var + "[" + PhGUtils::toString(idx) + "].diffuse";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << material.diffuse << endl;
    program->setUniformValue(str.c_str(), material.diffuse.toQVector());
    str = var + "[" + PhGUtils::toString(idx) + "].specular";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << material.specular << endl;
    program->setUniformValue(str.c_str(), material.specular.toQVector());
    str = var + "[" + PhGUtils::toString(idx) + "].ambient";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << material.ambient << endl;
    program->setUniformValue(str.c_str(), material.ambient.toQVector());
    str = var + "[" + PhGUtils::toString(idx) + "].shininess";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << material.shininess << endl;
    program->setUniformValue(str.c_str(), material.shininess);
    str = var + "[" + PhGUtils::toString(idx) + "].kcool";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << material.kcool << endl;
    program->setUniformValue(str.c_str(), material.kcool.toQVector());
    str = var + "[" + PhGUtils::toString(idx) + "].kwarm";
    //cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << material.kwarm << endl;
    program->setUniformValue(str.c_str(), material.kwarm.toQVector());

    // texture
    str = var + "[" + PhGUtils::toString(idx) + "].hasTexture";
    cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << hasTexture << endl;
    program->setUniformValue(str.c_str(), hasTexture);
    str = var + "[" + PhGUtils::toString(idx) + "].tex";
    cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << texId << endl;
    program->setUniformValue(str.c_str(), texId);

    // normal map
    str = var + "[" + PhGUtils::toString(idx) + "].hasNormalMap";
    cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << hasNormalMap << endl;
    program->setUniformValue(str.c_str(), hasNormalMap);
    str = var + "[" + PhGUtils::toString(idx) + "].nTex";
    cout << str << " @ " << program->uniformLocation(str.c_str()) << " = " << normalTexId << endl;
    program->setUniformValue(str.c_str(), normalTexId);

}

#ifndef SHAPE_H
#define SHAPE_H

#include "common.h"
#include "element.h"
#include "Geometry/point.hpp"
#include "Geometry/vector.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"

#include "Utils/stringutils.h"

#include <QImage>

struct Material {
    Material(){}
    Material(float3 diffuse, float3 specular, float3 ambient, float shininess, float3 kcool, float3 kwarm):
        diffuse(diffuse), specular(specular), ambient(ambient), shininess(shininess), kcool(kcool), kwarm(kwarm)
    {}
    float3 emission;
    float3 ambient;
    float3 diffuse;
    float3 specular;

    float shininess;

    float3 kcool;
    float3 kwarm;
};

struct Shape
{
    enum ShapeType {
        SPHERE = 0,
        PLANE = 1,
        ELLIPSOID = 2,
        CYLINDER,
        CONE,
        HYPERBOLOID,
        TRIANGLE_MESH
    };
    Shape(){}
    Shape(ShapeType t, float3 p, float r0, float r1, float r2,
          vec3f a0, vec3f a1, vec3f a2,
          Material mater):
    t(t), p(p), material(mater), hasTexture(false), texId(-1), hasNormalMap(false), normalTexId(-1) {
        axis[0] = a0; axis[1] = a1; axis[2] = a2;
        radius[0] = r0; radius[1] = r1; radius[2] = r2;

        vec3f ratio0 = a0/r0;
        vec3f ratio1 = a1/r1;
        vec3f ratio2 = a2/r2;
        m = PhGUtils::outerProduct<double, float>(ratio0, ratio0) + PhGUtils::outerProduct<double, float>(ratio1, ratio1) + PhGUtils::outerProduct<double, float>(ratio2, ratio2);
    }

    Shape(const Shape& s):t(s.t), p(s.p), angle(s.angle), m(s.m), height(s.height),
        material(s.material), hasTexture(s.hasTexture), texId(s.texId), hasNormalMap(s.hasNormalMap),
    normalTexId(s.normalTexId){
        for(int i=0;i<3;i++) {
            axis[i] = s.axis[i];
            radius[i] = s.radius[i];
        }
    }

    void uploadToShader(QGLShaderProgram *program, const string& var);
    void uploadToShader(QGLShaderProgram *program, const string& var, int idx);

    ShapeType t;

    // geometry
    float3 p;
    vec3f axis[3];
    float radius[3];
    float angle;
    mat3 m;
    float height;

    Material material;

    bool hasTexture;
    int texId;
    bool hasNormalMap;
    int normalTexId;
};

#endif // SHAPE_H

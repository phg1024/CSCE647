#ifndef SHAPE_H
#define SHAPE_H

#include <QColor>
#include "Geometry/point.hpp"
#include "Geometry/vector.hpp"

struct Shape
{
    Shape();

    QColor color;
    float Kd, Ks, Ka;
    float shininess;
};


struct Sphere : public Shape
{
    PhGUtils::Point3f center;
    float radius;
};

struct Ellipsoid : public Shape
{
    PhGUtils::Point3f center;
    PhGUtils::Vector3f axis[3];
    float radius[3];

};

struct Cylinder : public Shape
{
    PhGUtils::Point3f vertex;
    PhGUtils::Vector3f axis;
    float length;
    float radius;
};

struct Cone : public Shape
{
    PhGUtils::Point3f vertex;
    PhGUtils::Vector3f axis;
    float angle;
    float length;
};

struct Hyperboloid : public Shape
{
    // no idea
};

struct Plane : public Shape
{
    PhGUtils::Point3f point;
    PhGUtils::Vector3f normal;
    PhGUtils::Vector3f u, v;
    float w, h;
};

#endif // SHAPE_H

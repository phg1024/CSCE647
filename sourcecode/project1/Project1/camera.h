#ifndef CAMERA_H
#define CAMERA_H

#include "Geometry/point.hpp"
#include "Geometry/vector.hpp"

struct Camera
{
    typedef PhGUtils::Point3f point_t;
    typedef PhGUtils::Vector3f vector_t;

    Camera();

    point_t pos;
    vector_t up, dir;
    float f;
    float w, h;
};

#endif // CAMERA_H

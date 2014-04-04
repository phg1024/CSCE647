#ifndef CAMERA_H
#define CAMERA_H

#include "element.h"

struct Camera
{
    Camera();

    float3 pos;
    float3 up, dir;
    float f;
    float w, h;
};

#endif // CAMERA_H

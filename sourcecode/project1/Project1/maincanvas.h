#ifndef MAINCANVAS_H
#define MAINCANVAS_H

#include "OpenGL/gl3dcanvas.h"
#include "Geometry/Mesh.h"
#include "Geometry/MeshLoader.h"

class MainCanvas : public GL3DCanvas
{
public:
    MainCanvas();


protected:
    virtual void initialzeGL();
    virtual void paintGL();
    virtual void resizeGL(int w, int h);

private:
    PhGUtils::TriMesh mesh;
};

#endif // MAINCANVAS_H

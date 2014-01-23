#ifndef MAINCANVAS_H
#define MAINCANVAS_H

#include "OpenGL/gl3dcanvas.h"
#include <QGLShaderProgram>
#include <QGLShader>

class MainCanvas : public GL3DCanvas
{
public:
    MainCanvas(QWidget *parent = 0);
    ~MainCanvas();

protected:
    virtual void initializeGL();
    virtual void paintGL();
    virtual void resizeGL(int w, int h);

private:
    QGLShaderProgram* program;
    QGLShader* vShader, *fShader;
};

#endif // MAINCANVAS_H

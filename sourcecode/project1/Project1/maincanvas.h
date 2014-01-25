#ifndef MAINCANVAS_H
#define MAINCANVAS_H

#include "OpenGL/gl3dcanvas.h"
#include <QGLShaderProgram>
#include <QGLShader>

#include "camera.h"

class MainCanvas : public GL3DCanvas
{
public:
    MainCanvas(QWidget *parent = 0, QGLFormat format = qglformat_3d);
    ~MainCanvas();

protected:
    virtual void initializeGL();
    virtual void paintGL();
    virtual void resizeGL(int w, int h);

    virtual void keyPressEvent(QKeyEvent *e);
	virtual void mousePressEvent(QMouseEvent *e);
	virtual void mouseMoveEvent(QMouseEvent *e);

private:
    QGLShaderProgram* program;
    QGLShader* vShader, *fShader;

	bool renderingEnabled;
    int shadingMode;
	int samples;
	Camera cam;
};

#endif // MAINCANVAS_H

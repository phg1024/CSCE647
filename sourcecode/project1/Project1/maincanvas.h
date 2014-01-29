#ifndef MAINCANVAS_H
#define MAINCANVAS_H

#include "common.h"
#include "OpenGL/gl3dcanvas.h"

#include "scene.h"
#include "camera.h"
#include "light.h"

#include "Utils/Timer.h"

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

	void initShapes();
	void initLights();

	int loadTexture(const string& filename, int texSlot);

private:
    QGLShaderProgram* program;
    QGLShader* vShader, *fShader;

	bool renderingEnabled;
    int shadingMode;
	int samples;

	Scene scene;
};

#endif // MAINCANVAS_H

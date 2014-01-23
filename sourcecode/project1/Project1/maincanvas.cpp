#include "maincanvas.h"

MainCanvas::MainCanvas()
{
    PhGUtils::OBJLoader loader;
    loader.load("bunny.obj");

    mesh.initWithLoader(loader);
}

void MainCanvas::initialzeGL()
{
    GL3DCanvas::initializeGL();
}

void MainCanvas::resizeGL(int w, int h)
{
    GL3DCanvas::resizeGL(w, h);
}

void MainCanvas::paintGL()
{
    GL3DCanvas::paintGL();

    glClearColor(1, 1, 1, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor4f(0, 0, 0, 1);

    glPushMatrix();
    glScalef(1000, 1000, 1000);
    mesh.drawFrame();
    glPopMatrix();
}

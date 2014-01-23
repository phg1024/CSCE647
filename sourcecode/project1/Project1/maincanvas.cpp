#include "maincanvas.h"

MainCanvas::MainCanvas(QWidget* parent):
    GL3DCanvas(parent, qglformat_3d, ORTHONGONAL),
    program(nullptr),
    vShader(nullptr),
    fShader(nullptr)
{
    cout << "main canvas constructed." << endl;
    setSceneScale(1.0);
}

MainCanvas::~MainCanvas()
{
    delete program;
    delete vShader;
    delete fShader;
}

void MainCanvas::initializeGL()
{
    GL3DCanvas::initializeGL();

    program = new QGLShaderProgram(this);
    vShader = new QGLShader(QGLShader::Vertex);
    fShader = new QGLShader(QGLShader::Fragment);

    fShader->compileSourceFile("../Project1/frag.glsl");
    vShader->compileSourceFile("../Project1/vert.glsl");

    //program->addShader(vShader);
    program->addShader(fShader);

    program->link();
}

void MainCanvas::resizeGL(int w, int h)
{
    GL3DCanvas::resizeGL(w, h);
}

void MainCanvas::paintGL()
{
    // obtain the transform matrix from the trackball
    // apply the inverse transform to the camera


    if( program ) {
        program->bind();

        // upload scene structure
        program->setUniformValue("windowSize", QVector2D(width(), height()));

        glBegin(GL_QUADS);
        glVertex3f(-1.0, -1.0, 0.1);
        glVertex3f(1.0, -1.0, 0.1);
        glVertex3f(1.0, 1.0, 0.1);
        glVertex3f(-1.0, 1.0, 0.1);
        glEnd();

        program->release();
    }
}

#include "maincanvas.h"

MainCanvas::MainCanvas(QWidget* parent):
    GL3DCanvas(parent, qglformat_3d, ORTHONGONAL),
    program(nullptr),
    vShader(nullptr),
    fShader(nullptr)
{
    cout << "main canvas constructed." << endl;
    setSceneScale(1.0);
    shadingMode = 1;
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

void MainCanvas::keyPressEvent(QKeyEvent *e)
{
    GL3DCanvas::keyPressEvent(e);

    switch( e->key() ) {
    case Qt::Key_1:
    case Qt::Key_2:
    case Qt::Key_3:
    {
        int key = e->key() - Qt::Key_0;
        shadingMode = key;
        update();
        break;
    }
    default:
        break;
    }
}

void MainCanvas::paintGL()
{
    // obtain the transform matrix from the trackball
    // apply the inverse transform to the camera


    if( program ) {
        program->bind();

        // upload scene structure
        program->setUniformValue("windowSize", QVector2D(width(), height()));
        program->setUniformValue("lightCount", 2);
        program->setUniformValue("shadingMode", shadingMode);

        // for gooch shading
        program->setUniformValue("kcool", QVector3D(0, 0.1, 0.5));
        program->setUniformValue("kwarm", QVector3D(0.5, 0.5, 0.1));
        program->setUniformValue("kdiff", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("kspec", QVector3D(1, 1, 1));
        program->setUniformValue("kpos", QVector3D(-1, 0.5, -1));
        program->setUniformValue("alpha", 0.45f);
        program->setUniformValue("beta", 0.45f);

        glBegin(GL_QUADS);
        glVertex3f(-1.0, -1.0, 0.1);
        glVertex3f(1.0, -1.0, 0.1);
        glVertex3f(1.0, 1.0, 0.1);
        glVertex3f(-1.0, 1.0, 0.1);
        glEnd();

        program->release();
    }
}

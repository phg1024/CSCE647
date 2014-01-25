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
	samples = 1;

	renderingEnabled = true;

	cam.pos = Camera::point_t(0, 0, -5);
	cam.dir = Camera::vector_t(0, 0, 1);
	cam.up = Camera::vector_t(0, 1, 0);
	cam.f = 1.0;
	cam.w = 1.0;
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

	cout << "initializing GLEW ..." << endl;
	glewInit();

	cout << "loading shaders ..." << endl;
    program = new QGLShaderProgram(this);
    vShader = new QGLShader(QGLShader::Vertex);
    fShader = new QGLShader(QGLShader::Fragment);

    fShader->compileSourceFile("../Project1/frag.glsl");
	cout << qPrintable(fShader->log()) << endl;
    vShader->compileSourceFile("../Project1/vert.glsl");

    //program->addShader(vShader);
    program->addShader(fShader);

    program->link();
	cout << "init done." << endl;
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
	case Qt::Key_Space: 
	{
		/*
		shadingMode = shadingMode + 1;
		if( shadingMode > 3 ) shadingMode = 1;
		*/

		renderingEnabled = !renderingEnabled;
		update();
		break;
	}
	case Qt::Key_S:
	{
		cout << "Please input number of samples: " << endl;
		cin >> samples;
		cout << "Using " << samples << " samples." << endl;
		update();
		break;
	}
    default:
        break;
    }
}

void MainCanvas::paintGL()
{	
	if( !renderingEnabled ) return;
    // obtain the transform matrix from the trackball
    // apply the inverse transform to the camera
	CGLTrackball::elem_t* m = trackBall.getInverseMatrix();
	QMatrix4x4 mat(m);
	mat = mat.transposed();

	QVector3D camPos = cam.pos.toQVector();
	QVector3D camDir = cam.dir.toQVector();
	QVector3D camUp = cam.up.toQVector();

	camPos = (mat * QVector4D(camPos / trackBall.getScale(), 1.0)).toVector3D();
	camDir = (mat * QVector4D(camDir, 1.0)).toVector3D();
	camUp = (mat * QVector4D(camUp, 1.0)).toVector3D();

    if( program ) {
        program->bind();

        // upload scene structure
        program->setUniformValue("windowSize", QVector2D(width(), height()));
        program->setUniformValue("lightCount", 3);
        program->setUniformValue("shapeCount", 4);
        program->setUniformValue("shadingMode", shadingMode);
		program->setUniformValue("AAsamples", samples);

		// update camera info
		program->setUniformValue("camPos", camPos);
		program->setUniformValue("camUp", camUp);
		program->setUniformValue("camDir", camDir);
		program->setUniformValue("camF", cam.f);

        // for gooch shading
        program->setUniformValue("kdiff", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("kspec", QVector3D(1, 1, 1));
        program->setUniformValue("kpos", QVector3D(-1, 0.5, -1));
        program->setUniformValue("alpha", 0.15f);
        program->setUniformValue("beta", 0.25f);

		
        glBegin(GL_QUADS);
        glVertex3f(-1.0, -1.0, 0.1);
        glVertex3f(1.0, -1.0, 0.1);
        glVertex3f(1.0, 1.0, 0.1);
        glVertex3f(-1.0, 1.0, 0.1);
        glEnd();
		

        program->release();
    }
}

void MainCanvas::mousePressEvent(QMouseEvent *e)
{
	mouseState = DOWN;
	switch(mouseInteractionMode)
	{
	case VIEW_TRANSFORM:
		{
			switch(e->buttons() & 0xF)
			{
			case Qt::MidButton:
				{
					trackBall.mouse_translate(e->x(),e->y());
					break;
				}
			case Qt::LeftButton:
				{
					trackBall.mouse_rotate(e->x(), height() - e->y());
					break;
				}
			default:
				break;
			}            
			break;
		}
	case INTERACTION:
		break;
	}
	update();
}

void MainCanvas::mouseMoveEvent(QMouseEvent *e)
{
	switch(mouseInteractionMode)
	{
	case VIEW_TRANSFORM:
		{
			switch(e->buttons() & 0xF)
			{
			case Qt::MidButton:
				{
					trackBall.motion_translate(e->x(), e->y());
					update();
					break;
				}
			case Qt::LeftButton:
				{
					trackBall.motion_rotate(e->x(), height() - e->y());
					update();
					break;
				}
			default:
				break;
			}
			break;
		}
	case INTERACTION:
		break;
	}    
}

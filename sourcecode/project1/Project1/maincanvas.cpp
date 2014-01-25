#include "maincanvas.h"

MainCanvas::MainCanvas(QWidget* parent, QGLFormat format):
    GL3DCanvas(parent, format, ORTHONGONAL),
    program(nullptr),
    vShader(nullptr),
    fShader(nullptr)
{
    cout << "main canvas constructed." << endl;
    setSceneScale(1.0);
    shadingMode = 1;
    samples = 4;

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
    //delete vShader;
    delete fShader;
}

void MainCanvas::initializeGL()
{
    qDebug() << "OpenGL Versions Supported: " << QGLFormat::openGLVersionFlags();
    QString versionString(QLatin1String(reinterpret_cast<const char*>(glGetString(GL_VERSION))));
    qDebug() << "Driver Version String:" << versionString;
    qDebug() << "Current Context:" << this->format();

    GL3DCanvas::initializeGL();

	cout << "loading shaders ..." << endl;
    program = new QGLShaderProgram(this);
    vShader = new QGLShader(QGLShader::Vertex);
    fShader = new QGLShader(QGLShader::Fragment);

    fShader->compileSourceFile("../Project1/frag.glsl");
	cout << qPrintable(fShader->log()) << endl;
    vShader->compileSourceFile("../Project1/vert.glsl");
    cout << qPrintable(vShader->log()) << endl;

    program->addShader(vShader);
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

        // upload scene parameters
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

        // set up ray tracing parameters
        program->setUniformValue("background.t", -1.0f);
        program->setUniformValue("background.color", QVector3D(0.85, .85, .85));

        // upload object information
        program->setUniformValue("shapes[0].type", 0);
        program->setUniformValue("shapes[0].p", QVector3D(0, 0, 1));
        program->setUniformValue("shapes[0].radius[0]", 1.0f);
        program->setUniformValue("shapes[0].diffuse", QVector3D(0.25, 0.5, 1.0));
        program->setUniformValue("shapes[0].specular", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("shapes[0].ambient", QVector3D(0.05, 0.10, 0.15));
        program->setUniformValue("shapes[0].shininess", 50.0f);
        program->setUniformValue("shapes[0].kcool", QVector3D(0, 0, .4));
        program->setUniformValue("shapes[0].kwarm", QVector3D(.4, .4, 0));

        program->setUniformValue("shapes[1].type", 1);
        program->setUniformValue("shapes[1].p", QVector3D(0, -1, 0));
        program->setUniformValue("shapes[1].axis[0]", QVector3D(0, 1.0, 0));
        program->setUniformValue("shapes[1].axis[1]", QVector3D(1.0, 0, 0));
        program->setUniformValue("shapes[1].axis[2]", QVector3D(0, 0, 1.0));
        program->setUniformValue("shapes[1].radius[0]", 3.0f);
        program->setUniformValue("shapes[1].radius[1]", 6.0f);
        program->setUniformValue("shapes[1].diffuse", QVector3D(0.75, 0.75, 0.75));
        program->setUniformValue("shapes[1].specular", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("shapes[1].ambient", QVector3D(0.05, 0.05, 0.05));
        program->setUniformValue("shapes[1].shininess", 50.0f);
        program->setUniformValue("shapes[1].kcool", QVector3D(0, 0, .4));
        program->setUniformValue("shapes[1].kwarm", QVector3D(.4, .4, 0));

        program->setUniformValue("shapes[2].type", 0);
        program->setUniformValue("shapes[2].p", QVector3D(-0.5, 0.5, -1));
        program->setUniformValue("shapes[2].radius[0]", 0.25f);
        program->setUniformValue("shapes[2].diffuse", QVector3D(0.25, 0.75, 0.25));
        program->setUniformValue("shapes[2].specular", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("shapes[2].ambient", QVector3D(0.05, 0.05, 0.05));
        program->setUniformValue("shapes[2].shininess", 20.0f);
        program->setUniformValue("shapes[2].kcool", QVector3D(0, 0.4, 0));
        program->setUniformValue("shapes[2].kwarm", QVector3D(.4, 0, .4));

        program->setUniformValue("shapes[3].type", 0);
        program->setUniformValue("shapes[3].p", QVector3D(0.75, -0.5, -0.5));
        program->setUniformValue("shapes[3].radius[0]", 0.5f);
        program->setUniformValue("shapes[3].diffuse", QVector3D(0.75, 0.75, 0.25));
        program->setUniformValue("shapes[3].specular", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("shapes[3].ambient", QVector3D(0.05, 0.05, 0.05));
        program->setUniformValue("shapes[3].shininess", 100.0f);
        program->setUniformValue("shapes[3].kcool", QVector3D(.9, .1, .6));
        program->setUniformValue("shapes[3].kwarm", QVector3D(.05, .45, .05));

        // setup lights
        program->setUniformValue("lights[0].intensity", 0.75f);
        program->setUniformValue("lights[0].ambient", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[0].diffuse", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[0].specular", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[0].pos", QVector3D(-2.0, 2.0, -10.0));

        program->setUniformValue("lights[1].intensity", 0.25f);
        program->setUniformValue("lights[1].ambient", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[1].diffuse", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[1].specular", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[1].pos", QVector3D(4.0, 4.0, -10.0));

        program->setUniformValue("lights[2].intensity", 0.25f);
        program->setUniformValue("lights[2].ambient", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[2].diffuse", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[2].specular", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("lights[2].pos", QVector3D(0.0, 1.0, -10.0));

        // for gooch shading
        program->setUniformValue("kdiff", QVector3D(1.0, 1.0, 1.0));
        program->setUniformValue("kspec", QVector3D(1, 1, 1));
        program->setUniformValue("kpos", QVector3D(-1, 0.5, -1));
        program->setUniformValue("alpha", 0.15f);
        program->setUniformValue("beta", 0.25f);

        glColor4f(0.2, 0.3, 0.5, 1.0);
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

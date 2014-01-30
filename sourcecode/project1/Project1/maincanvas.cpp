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

	scene.cam.pos = float3(0, 0, -5);
	scene.cam.dir = float3(0, 0, 1);
	scene.cam.up = float3(0, 1, 0);
	scene.cam.f = 1.0;
	scene.cam.w = 1.0;
}

MainCanvas::~MainCanvas()
{
    delete program;
    delete vShader;
    delete fShader;
}

void MainCanvas::initializeGL()
{
    qDebug() << "OpenGL Versions Supported: " << QGLFormat::openGLVersionFlags();
    QString versionString(QLatin1String(reinterpret_cast<const char*>(glGetString(GL_VERSION))));
    qDebug() << "Driver Version String:" << versionString;
    qDebug() << "Current Context:" << this->format();

    GL3DCanvas::initializeGL();

	makeCurrent();

	cout << "loading shaders ..." << endl;
    program = new QGLShaderProgram(this);
    vShader = new QGLShader(QGLShader::Vertex);
    fShader = new QGLShader(QGLShader::Fragment);

	PhGUtils::Timer t;
	t.tic();
    fShader->compileSourceFile("../Project1/frag.glsl");
	cout << qPrintable(fShader->log()) << endl;
    vShader->compileSourceFile("../Project1/vert.glsl");
    cout << qPrintable(vShader->log()) << endl;

    program->addShader(vShader);
    program->addShader(fShader);
	t.toc("shader compilation");

	t.tic();
    program->link();
	t.toc("shader linking");
	cout << "done." << endl;

	cout << "initializing scene ..." << endl;
	initLights();
	initShapes();
	cout << "done." << endl;

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
		cout << e->key() << endl;
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

	QVector3D camPos = scene.cam.pos.toQVector();
	QVector3D camDir = scene.cam.dir.toQVector();
	QVector3D camUp = scene.cam.up.toQVector();

	camPos = (mat * QVector4D(camPos / trackBall.getScale(), 1.0)).toVector3D();
	camDir = (mat * QVector4D(camDir, 1.0)).toVector3D();
	camUp = (mat * QVector4D(camUp, 1.0)).toVector3D();

    if( program ) {
        program->bind();

        // upload scene parameters
        program->setUniformValue("windowSize", QVector2D(width(), height()));
        program->setUniformValue("lightCount", (int)scene.lights.size());
        program->setUniformValue("shapeCount", (int)scene.shapes.size());
        program->setUniformValue("shadingMode", shadingMode);
		program->setUniformValue("AAsamples", samples);

		// update camera info
		program->setUniformValue("camPos", camPos);
		program->setUniformValue("camUp", camUp);
		program->setUniformValue("camDir", camDir);
		program->setUniformValue("camF", scene.cam.f);

        // set up ray tracing parameters
        program->setUniformValue("background.t", -1.0f);
        program->setUniformValue("background.color", QVector3D(0.85f, .85f, .85f));

        // upload object information
        for(int idx=0;idx<scene.shapes.size();idx++) {
			scene.shapes[idx].uploadToShader(program, "shapes", idx);
		}

        // setup lights
		for(int idx=0;idx<scene.lights.size();idx++) {
			scene.lights[idx].uploadToShader(program, "lights", idx);
		}

        // for gooch shading
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
					trackBall.mouse_rotate(width() - e->x(), height() - e->y());
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
					trackBall.motion_rotate(width() - e->x(), height() - e->y());
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

void MainCanvas::initShapes()
{
	scene.shapes.clear();

	
	Shape s0( Shape::SPHERE, 
		float3(0, 0, 1),	// p 
		1.0, 0.0, 0.0,		// radius
		vec3f(),			// axis[0]
		vec3f(),			// axis[1]
		vec3f(),			// axis[2]
		Material(
		float3(1.0, 1.0, 1.0),		// diffuse
		float3(1.0, 1.0, 1.0),		// specular
		float3(0.10, 0.10, 0.1),		// ambient
		50.0f,							// shininess
		float3(0, 0, .4),				// kcool
		float3(.4, .4, 0)				// kwarm
		)
	);
	s0.hasTexture = true;
	s0.texId = loadTexture("textures/earth/earthmap4k.png", 0);
	s0.hasNormalMap = true;
	s0.normalTexId = loadTexture("textures/earth/earth_normalmap_flat_4k.png", 1);
	//s0.texId = loadTexture("textures/gabby.jpg", 0);
	scene.shapes.push_back(s0);
	

	Shape s(
		Shape::PLANE,
		float3(0, -1, 0),
		3.0, 3.0, 0.0,
		vec3f(0, 1, 0),
		vec3f(1, 0, 0),
		vec3f(0, 0, 1),
		Material(
		float3(0.75, 0.75, 0.75),
		float3(1, 1, 1),
		float3(0.05, 0.05, 0.05),
		50.0,
		float3(0, 0, .4),
		float3(.4, .4, 0)
		));
	s.hasTexture = true;
	s.texId = loadTexture("chessboard.png", 2);
	scene.shapes.push_back(s);
	
	scene.shapes.push_back(Shape( Shape::SPHERE, 
		float3(0, 0, 1),	// p 
		1.0, 0.0, 0.0,		// radius
		vec3f(),			// axis[0]
		vec3f(),			// axis[1]
		vec3f(),			// axis[2]
		Material(
		float3(0.25, 0.5 , 1.0 ),		// diffuse
		float3(1.0 , 1.0 , 1.0 ),		// specular
		float3(0.05, 0.10, 0.15),		// ambient
		50.0f,							// shininess
		float3(0, 0, .4),				// kcool
		float3(.4, .4, 0)				// kwarm
		)
		)
		);

	Shape s2( Shape::SPHERE, 
		float3(-0.5, 0.5, -1),	// p 
		0.25, 0.0, 0.0,		// radius
		vec3f(),			// axis[0]
		vec3f(),			// axis[1]
		vec3f(),			// axis[2]
		Material(
		float3(0.75, 0.75, 0.75),		// diffuse
		float3(1.0 , 1.0 , 1.0),		// specular
		float3(0.05, 0.05, 0.05),		// ambient
		20.0f,							// shininess
		float3(0, .4, 0),				// kcool
		float3(.4, 0, .4)				// kwarm
		));
	s2.hasTexture = true;
	s2.texId = loadTexture("textures/moon/moon_map_4k.png", 3);
	s2.hasNormalMap = true;
	s2.normalTexId = loadTexture("textures/moon/moon_normal_4k.png", 4);
	scene.shapes.push_back(s2);

	scene.shapes.push_back(Shape( Shape::ELLIPSOID, 
		float3(1.0, -0.5, -0.5),	// p 
		0.75, 0.25, 0.25,		// radius
		vec3f(1, 0, 1),			// axis[0]
		vec3f(1, 1, 0),			// axis[1]
		vec3f(0, 1, 1),			// axis[2]
		Material(
		float3(0.75, 0.75, 0.25),		// diffuse
		float3(1.0 , 1.0 , 1.0),		// specular
		float3(0.05, 0.05, 0.05),		// ambient
		100.0f,							// shininess
		float3(.9, .1, .6),				// kcool
		float3(.05, .45, .05)				// kwarm
		)
		)
		);
	
	scene.shapes.push_back(Shape( Shape::CYLINDER, 
		float3(-1.0, -0.5, 0.5),	// p 
		0.5, 1.0, 0.25,		// radius
		vec3f(0, 1, 0),			// axis[0]
		vec3f(1, 1, 0),			// axis[1]
		vec3f(0, 1, 1),			// axis[2]
		Material(
		float3(0.75, 0.75, 0.25),		// diffuse
		float3(1.0 , 1.0 , 1.0),		// specular
		float3(0.05, 0.05, 0.05),		// ambient
		100.0f,							// shininess
		float3(.9, .1, .6),				// kcool
		float3(.05, .45, .05)				// kwarm
		)
		)
		);
	
	scene.shapes.push_back(Shape( Shape::CONE, 
		float3(0.5, -0.5, -1.0),	// p 
		0.25, 1.0, 0.8,		// radius
		vec3f(0, 1, 0),			// axis[0]
		vec3f(1, 1, 0),			// axis[1]
		vec3f(0, 1, 1),			// axis[2]
		Material(
		float3(0.75, 0.75, 0.25),		// diffuse
		float3(1.0 , 1.0 , 1.0),		// specular
		float3(0.05, 0.05, 0.05),		// ambient
		100.0f,							// shininess
		float3(.9, .1, .6),				// kcool
		float3(.05, .45, .05)				// kwarm
		)
		)
		);
}

void MainCanvas::initLights()
{
	scene.lights.clear();

	scene.lights.push_back(Light(Light::POINT, 0.75, float3(1, 1, 1), float3(1, 1, 1), float3(1, 1, 1), float3(-2, 4, -10)));
	scene.lights.push_back(Light(Light::POINT, 0.25, float3(1, 1, 1), float3(1, 1, 1), float3(1, 1, 1), float3(4, 4, -10)));
	scene.lights.push_back(Light(Light::DIRECTIONAL, 0.25, float3(1, 1, 1), float3(1, 1, 1), float3(1, 1, 1), float3(0, 10, -10), float3(0, -10/sqrtf(200.f), 10/sqrtf(200.f))));
}

int MainCanvas::loadTexture(const string& filename, int texSlot)
{
	QImage img(filename.c_str());
	cout << img.width() << "x" << img.height() << endl;

	unsigned char* data = new unsigned char[img.width()*img.height()*4];
	for(int i=0;i<img.height();i++) {
		for(int j=0;j<img.width();j++) {
			int idx=(i*img.width()+j)*4;
			QRgb pix = img.pixel(j, i);

			data[idx] = (unsigned char)qRed(pix);
			data[idx+1] = (unsigned char)qGreen(pix);
			data[idx+2] = (unsigned char)qBlue(pix);
			data[idx+3] = (unsigned char)qAlpha(pix);
		}
	}

	GLuint texture;
	glGenTextures( 1, &texture ); //generate the texture with the loaded data
	glActiveTexture (GL_TEXTURE0 + texSlot);
	glBindTexture( GL_TEXTURE_2D, texture ); //bind the texture to its array
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE ); //set texture environment parameters

	//And if you go and use extensions, you can use Anisotropic filtering textures which are of an
	//even better quality, but this will do for now.
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
		GL_LINEAR);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, 
		GL_LINEAR);

	//Here we are setting the parameter to repeat the texture instead of clamping the texture
	//to the edge of our shape. 
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );

	//Generate the texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width(), img.height(), 0, GL_RGBA, 
		GL_UNSIGNED_BYTE, data);

	delete[] data;
	return texSlot; //return whether it was successful
}

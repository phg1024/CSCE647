#include "maincanvas.h"
#include "Utils/Timer.h"

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

    glewExperimental = true;
    glewInit();

    int x;
    glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &x);
    cout << "max frag uniform comp = " << x << endl;

    initLights();
    initShapes();
    initVBOs();

    cout << "constructing shaders ..." << endl;
    program = new QGLShaderProgram(this);
    vShader = new QGLShader(QGLShader::Vertex);
    fShader = new QGLShader(QGLShader::Fragment);

    string fragStr = buildFragmentShaderSourceCode();
    string vertStr = buildVertexShaderSourceCode();

    cout << fragStr << endl;
    fShader->compileSourceCode(fragStr.c_str());
    //fShader->compileSourceFile("../Project1/frag.glsl");
    cout << qPrintable(fShader->log()) << endl;

    cout << vertStr << endl;
    vShader->compileSourceCode(vertStr.c_str());
    //vShader->compileSourceFile("../Project1/vert.glsl");
    cout << qPrintable(vShader->log()) << endl;

    program->addShader(vShader);
    program->addShader(fShader);

    program->link();

    cout << "init done." << endl;

//    cout << "pause" << endl;
//    char dummy;
//    cin >> dummy;
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

    PhGUtils::Timer t;
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

//		// update camera info
        program->setUniformValue("camPos", camPos);
        program->setUniformValue("camUp", camUp);
        program->setUniformValue("camDir", camDir);
        program->setUniformValue("camF", scene.cam.f);

//        // set up ray tracing parameters
        program->setUniformValue("background.t", -1.0f);
        program->setUniformValue("background.color", QVector3D(0.85f, .85f, .85f));

        // upload object information
        for(int idx=0;idx<scene.shapes.size();idx++) {
            scene.shapes[idx].uploadToShader(program, "shapes", idx);
            string str;
            str = "textures[" + PhGUtils::toString(scene.shapes[idx].texId) + "]";
            program->setUniformValue(str.c_str(), scene.shapes[idx].texId);
            str = "textures[" + PhGUtils::toString(scene.shapes[idx].normalTexId) + "]";
            program->setUniformValue(str.c_str(), scene.shapes[idx].normalTexId);
        }

        // upload light information
        for(int idx=0;idx<scene.lights.size();idx++) {
            scene.lights[idx].uploadToShader(program, "lights", idx);
        }

        GLuint64 elapsed;
        GLuint qID;
        glGenQueries(1, &qID);

        glBeginQuery(GL_TIME_ELAPSED, qID);

        glBindVertexArray (vao);
        // draw points 0-3 from the currently bound VAO with current in-use shader
        glDrawArrays (GL_TRIANGLE_FAN, 0, 4);

        glEndQuery(GL_TIME_ELAPSED);

        GLint stopTimerAvailable = 0;
        while (!stopTimerAvailable) {
            glGetQueryObjectiv(qID, GL_QUERY_RESULT_AVAILABLE, &stopTimerAvailable);
        }

        // get query results
        glGetQueryObjectui64v(qID, GL_QUERY_RESULT, &elapsed);

        printf("Time spent on the GPU: %f ms\n", elapsed / 1000000.0);

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
        1.0, 1.0, 1.0,		// radius
        vec3f(1, 0, 0),			// axis[0]
        vec3f(0, 1, 0),			// axis[1]
        vec3f(0, 0, 1),			// axis[2]
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
    s0.texId = loadTexture("textures/earth/earthmap2k.png", 0);
    s0.hasNormalMap = true;
    s0.normalTexId = loadTexture("textures/earth/earth_normalmap_flat_2k.png", 1);
    scene.shapes.push_back(s0);

    Shape s(
        Shape::PLANE,
        float3(0, -1, 0),
        3.0, 3.0, 3.0,
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
        1.0, 1.0, 1.0,		// radius
        vec3f(1, 0, 0),			// axis[0]
        vec3f(0, 1, 0),			// axis[1]
        vec3f(0, 0, 1),			// axis[2]
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
        0.25, 0.25, 0.25,		// radius
        vec3f(1, 0, 0),			// axis[0]
        vec3f(0, 1, 0),			// axis[1]
        vec3f(0, 0, 1),			// axis[2]
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
}

void MainCanvas::initLights()
{
    scene.lights.clear();

    scene.lights.push_back(Light(Light::POINT, 0.75, float3(1, 1, 1), float3(1, 1, 1), float3(1, 1, 1), float3(-2, 4, -10)));
    scene.lights.push_back(Light(Light::POINT, 0.25, float3(1, 1, 1), float3(1, 1, 1), float3(1, 1, 1), float3(4, 4, -10)));
    scene.lights.push_back(Light(Light::DIRECTIONAL, 0.25, float3(1, 1, 1), float3(1, 1, 1), float3(1, 1, 1), float3(0, 10, -10), float3(0, -10/sqrtf(200.f), 10/sqrtf(200.f))));
}

void MainCanvas::initVBOs()
{
    float points[] = {
       -1.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  0.0f,
        1.0f, -1.0f,  0.0f,
       -1.0f, -1.0f,  0.0f
    };

    // vertex buffer object
    vbo = 0;
    // create buffer
    glGenBuffers (1, &vbo);
    // bind the buffer
    glBindBuffer (GL_ARRAY_BUFFER, vbo);
    // bind data to the buffer
    glBufferData (GL_ARRAY_BUFFER, 12 * sizeof (float), points, GL_STATIC_DRAW);

    // vertex attribute object
    vao = 0;
    // create attr object
    glGenVertexArrays (1, &vao);
    // bind the object
    glBindVertexArray (vao);
    // enable it
    glEnableVertexAttribArray (0);
    // bind buffer to current vao
    glBindBuffer (GL_ARRAY_BUFFER, vbo);
    // specify vao specs
    glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
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

unsigned long getFileLength(ifstream& file)
{
    if(!file.good()) return 0;

    unsigned long pos=file.tellg();
    file.seekg(0,ios::end);
    unsigned long len = file.tellg();
    file.seekg(ios::beg);

    return len;
}

string readFileAsString(const string &filename)
{
    ifstream file;
    file.open(filename, ios::in); // opens as ASCII!
    if(!file) return string();

    int shaderLength = getFileLength(file);

    if (shaderLength==0) return string();   // Error: Empty File

    char* shaderSource = new char[shaderLength+1];
    if (shaderSource == 0) return string();   // can't reserve memory

     // len isn't always strlen cause some characters are stripped in ascii read...
     // it is important to 0-terminate the real length later, len is just max possible value...
    shaderSource[shaderLength] = 0;

    unsigned int i=0;
    while (file.good())
    {
        shaderSource[i] = file.get();       // get character from file.
        if (!file.eof())
         i++;
    }

    shaderSource[i] = 0;  // 0-terminate it at the correct position

    file.close();

    return string(shaderSource);
}

string MainCanvas::buildFragmentShaderSourceCode()
{
    //return readFileAsString("../Project1/shaders/frag3.glsl");

    string versionTag = "#version 330";
    string defStr = readFileAsString("../Project1/shaders/definitions.glsl");
    string varStr = readFileAsString("../Project1/shaders/variables.glsl");

    //string shapeStr = scene.createShapesSourceCode();
    //string lightStr = scene.createLightSourceCode();

    string utilStr = readFileAsString("../Project1/shaders/utils.glsl");
    string initStr = readFileAsString("../Project1/shaders/initialize.glsl");
    //initStr += scene.createInitializationSourceCode();
    string rayStr = readFileAsString("../Project1/shaders/rays.glsl");

    string intStr = readFileAsString("../Project1/shaders/intersectionTests.glsl");
    string shadingStr = readFileAsString("../Project1/shaders/shading.glsl");
    string tracingStr = readFileAsString("../Project1/shaders/rayTracing.glsl");
    string fragStr = readFileAsString("../Project1/shaders/frag.glsl");


    return versionTag + "\n"
         + defStr + "\n"
         + varStr + "\n"
         //+ shapeStr + "\n"
         //+ lightStr + "\n"
         + utilStr + "\n"
         + initStr + "\n"
         + rayStr + "\n"
         + intStr + "\n"
         + shadingStr + "\n"
         + tracingStr + "\n"
         + fragStr;
}

string MainCanvas::buildVertexShaderSourceCode()
{
    return readFileAsString("../Project1/shaders/vert.glsl");
}

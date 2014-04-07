#pragma once

// OpenGL related
#include "GL/glew.h"
#ifdef WIN32
#include "GL/freeglut.h"
#else
#ifdef __APPLE__
#include "GLUT/glut.h"
#else
#include "GL/glut.h"
#endif
#endif

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath>
using namespace std;

#include <QApplication>
#include <QGLShaderProgram>
#include <QGLShader>

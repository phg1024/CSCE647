#-------------------------------------------------
#
# Project created by QtCreator 2014-01-23T00:17:04
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Project1
TEMPLATE -= app

INCLUDEPATH += $$PWD/../../../../../../Utils/glew-1.10.0/include
LIBS += -L$$PWD/../../../../../../Utils/glew-1.10.0/lib -lGLEW
LIBS += -framework GLUT

CONFIG -= app_bundle
CONFIG += c++11

SOURCES += main.cpp\
        mainwindow.cpp \
    maincanvas.cpp \
    camera.cpp \
    shape.cpp \
    light.cpp \
    scene.cpp

HEADERS  += mainwindow.h \
    maincanvas.h \
    camera.h \
    shape.h \
    light.h \
    common.h \
    element.h \
    scene.h

FORMS    += mainwindow.ui


OTHER_FILES += \
    shaders/definitions.glsl \
    shaders/frag.glsl \
    shaders/initialize.glsl \
    shaders/intersectionTests.glsl \
    shaders/rays.glsl \
    shaders/rayTracing.glsl \
    shaders/shading.glsl \
    shaders/utils.glsl \
    shaders/variables.glsl \
    shaders/vert.glsl \
    shaders/shading_simple.glsl \
    shaders/intersectionTests_simple.glsl

unix: LIBS += -L$$PWD/../../../../../../Utils/PhGLib/lib/ -lPhGLib

INCLUDEPATH += $$PWD/../../../../../../Utils/PhGLib/include
DEPENDPATH += $$PWD/../../../../../../Utils/PhGLib/include

unix: PRE_TARGETDEPS += $$PWD/../../../../../../Utils/PhGLib/lib/libPhGLib.a

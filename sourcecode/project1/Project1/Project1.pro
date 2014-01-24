#-------------------------------------------------
#
# Project created by QtCreator 2014-01-23T00:17:04
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Project1
TEMPLATE -= app

QMAKE_CXXFLAGS += -std=c++0x -stdlib=libc++ -mmacosx-version-min=10.8

LIBS += -lm -lc++ -framework OpenGL -framework GLUT -lGLEW

CONFIG-=app_bundle

SOURCES += main.cpp\
        mainwindow.cpp \
    maincanvas.cpp \
    camera.cpp \
    shape.cpp

HEADERS  += mainwindow.h \
    maincanvas.h \
    camera.h \
    shape.h

FORMS    += mainwindow.ui


OTHER_FILES += \
    frag.glsl \
    vert.glsl

unix: LIBS += -L$$PWD/../../../../../Documents/PhGLib/lib/ -lPhGLib

INCLUDEPATH += $$PWD/../../../../../Documents/PhGLib/include
DEPENDPATH += $$PWD/../../../../../Documents/PhGLib/include

unix: PRE_TARGETDEPS += $$PWD/../../../../../Documents/PhGLib/lib/libPhGLib.a

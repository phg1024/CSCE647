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

LIBS += -lm -lc++

CONFIG-=app_bundle

SOURCES += main.cpp\
        mainwindow.cpp \
    maincanvas.cpp

HEADERS  += mainwindow.h \
    maincanvas.h

FORMS    += mainwindow.ui

unix: LIBS += -L$$PWD/../../../../../../Documents/Codes/build-PhGLib-Qt_4_8_4_Clang-Release/ -lPhGLib

INCLUDEPATH += $$PWD/../../../../../../Documents/Codes/PhGLib/include
DEPENDPATH += $$PWD/../../../../../../Documents/Codes/PhGLib/include

unix: PRE_TARGETDEPS += $$PWD/../../../../../../Documents/Codes/build-PhGLib-Qt_4_8_4_Clang-Release/libPhGLib.a

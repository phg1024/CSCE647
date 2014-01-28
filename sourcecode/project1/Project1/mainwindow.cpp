#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QGLFormat>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QGLFormat format;
    format.setVersion( 3, 2 );
    format.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
    format.setOption(QGL::DoubleBuffer       |
                     QGL::DepthBuffer        |
                     QGL::AccumBuffer        |
                     //QGL::StencilBuffer      |
                     //QGL::StereoBuffers      |
                     QGL::SampleBuffers      |
                     QGL::Rgba               |
                     QGL::AlphaChannel       |
                     QGL::DirectRendering    |
                     QGL::HasOverlay);

    //QGLFormat::setDefaultFormat(format);

    canvas = new MainCanvas(parent, format);
    //canvas = new MainCanvas(parent);

    this->setCentralWidget(canvas);
}

MainWindow::~MainWindow()
{
    delete ui;
}

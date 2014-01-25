#include "mainwindow.h"
#include <QApplication>

#ifdef WIN32
#include "Utils/console.h"
#endif

int main(int argc, char *argv[])
{
#ifdef WIN32
	createConsole();
#endif

    QApplication a(argc, argv);

    MainWindow w;
    w.show();
    return a.exec();
}

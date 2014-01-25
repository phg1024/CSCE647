#include "mainwindow.h"
#include <QApplication>

#include "Utils/console.h"

int main(int argc, char *argv[])
{
	createConsole();

    QApplication a(argc, argv);

    MainWindow w;
    w.show();
    return a.exec();
}

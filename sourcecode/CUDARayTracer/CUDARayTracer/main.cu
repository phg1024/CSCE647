// includes, system
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iostream>
using namespace std;

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include "GLFWWindowManager.h"
#include "RayTracerWindow.h"
#include "FreeImagePlus.h"

int main(int argc, char **argv)
{
	FreeImage_Initialise();

	srand(clock());
	
	CUDARayTracer raytracer;
	raytracer.init();
	raytracer.run();
	
	FreeImage_DeInitialise();
	return 0;
}

#include "GLFWWindowManager.h"
#include "FreeImagePlus.h"
#include "extras/hdr/rgbe.h"


GLFWWindow::GLFWWindow(int w, int h, const string& title):
	w(w), h(h), title(title)
{
	init();
}

GLFWWindow::~GLFWWindow(void) {
	cout << "destroying " << title << endl;
	if( window != nullptr ) destroy();
	cout << "done." << endl;
}

map<GLFWwindow*, GLFWWindow*> GLFWWindowManager::windowmap;
GLFWWindowManager* GLFWWindowManager::manager = nullptr;

void GLFWWindow::makeCurrent() const {
	glfwMakeContextCurrent(window);
}

void GLFWWindow::doneCurrent() const {
	glfwMakeContextCurrent(NULL);
}

void GLFWWindow::destroy() {
	glfwDestroyWindow(window);
	window = nullptr;
}

bool GLFWWindow::init() {
	if( !glfwInit() ) {
		return false;
	}
	
	window = glfwCreateWindow(w, h, title.c_str(), NULL, NULL);
	if( !window ) {
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(window);

	glewInit();
	if ( !glewIsSupported("GL_VERSION_2_0 ") )
	{
		cerr << "ERROR: Support for necessary OpenGL extensions missing." << endl;
		return false;
	}

		// Print out GLFW, OpenGL version and GLEW Version:
	int iOpenGLMajor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
	int iOpenGLMinor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
	int iOpenGLRevision = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
	printf("Status: Using GLFW Version %s\n", glfwGetVersionString());
	printf("Status: Using OpenGL Version: %i.%i, Revision: %i\n", iOpenGLMajor, iOpenGLMinor, iOpenGLRevision);
	printf("Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	return true;
}

void GLFWWindow::display() {
	makeCurrent();

	glutSolidTeapot(1.0);

	doneCurrent();
}

void GLFWWindow::resize(int width, int height) {
	w = width; h = height;
	// viewport
	glViewport(0, 0, w, h);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, w, 0, h, 0.1, 10.0);
}

void GLFWWindow::keyboard(int key, int scancode, int action, int mods) {
}


void GLFWWindow::mouse(int button, int action, int mods) {
}


void GLFWWindow::cursor_pos(double x, double y) {
}

void GLFWWindow::screenshot(const string& filename) {
	int width = w;
	int height = h;
	cout << "canvas size = " << w << "x" << h << endl;
	BYTE* pixels = new BYTE[ 3 * width * height];
	glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);

	FIBITMAP* bitmap = FreeImage_Allocate (width, height, 24);
	RGBQUAD color ;
	if (! bitmap ) {
		cerr << "Failed to save screenshot.png." << endl;
	}

	for ( int i=0, idx=0; i<height; i++) { 
		for ( int j=0; j<width; j++) {
			color.rgbBlue = pixels[idx++];
			color.rgbGreen = pixels[idx++];
			color.rgbRed = pixels[idx++];
			FreeImage_SetPixelColor ( bitmap, j, i, &color );
		}
	}
		
	if ( FreeImage_Save (FIF_PNG, bitmap , "screenshot.png" , 0) )
		cout << "screenshot.png saved!" << endl;
}


GLFWWindow* GLFWWindowManager::createWindow(int w, int h, const string& title) {
	GLFWWindow* window = new GLFWWindow(w, h, title);
	registerWindow(window);
	return window;
}

void GLFWWindowManager::registerWindow(GLFWWindow* window) {
	auto handle = window->getHandle();
	windowmap[handle] = window;

	glfwSetKeyCallback(handle, keyboard_callback);
	glfwSetMouseButtonCallback(handle, mouse_callback);
	glfwSetCursorPosCallback(handle, cursorpos_callback);
	glfwSetWindowSizeCallback(handle, resize_callback);	
}

bool GLFWWindowManager::shouldClose() {
	if( windowmap.empty() ) return true;

	std::list<GLFWWindow*> lToDelete;
	for (const auto& x : windowmap)
	{
		if (glfwWindowShouldClose(x.first))
		{
			lToDelete.push_back(x.second);
		}
	}

	if (!lToDelete.empty())
	{
		// we have windows to delete, Delete them:
		for (GLFWWindow* x : lToDelete)
		{			

			auto handle = x->getHandle();
			// destroy the window
			x->destroy();

			delete x;

			// unregister the window
			windowmap.erase(windowmap.find(handle));
		}
	}

	if (windowmap.empty())
		return true;

	return false;
}

void GLFWWindowManager::run() {
	while( !shouldClose() ) {
		for( const auto& x : windowmap ) {
			x.second->makeCurrent();
			
			x.second->display();

			glfwSwapBuffers(x.first);
			x.second->doneCurrent();
		}

		glfwPollEvents();
	}
}

GLFWWindow* GLFWWindowManager::getHandle(GLFWwindow* window) {
	if( windowmap.find(window) != windowmap.end() ) return windowmap[window];
	else return nullptr;
}

void GLFWWindowManager::keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	GLFWWindow* manager = getHandle(window);
	if( manager != nullptr ) manager->keyboard(key, scancode, action, mods);
	else {
		cerr << "Invalid window!" << endl;
	}
}

void GLFWWindowManager::mouse_callback(GLFWwindow* window, int button, int action, int mods) {
	GLFWWindow* manager = getHandle(window);
	if( manager != nullptr ) manager->mouse(button, action, mods);
	else {
		cerr << "Invalid window!" << endl;
	}
}

void GLFWWindowManager::cursorpos_callback(GLFWwindow* window, double x, double y) {
	GLFWWindow* manager = getHandle(window);
	if( manager != nullptr ) manager->cursor_pos(x, y);
	else {
		cerr << "Invalid window!" << endl;
	}
}

void GLFWWindowManager::resize_callback(GLFWwindow* window, int width, int height) {
	GLFWWindow* manager = getHandle(window);
	if( manager != nullptr ) manager->resize(width, height);
	else {
		cerr << "Invalid window!" << endl;
	}
}


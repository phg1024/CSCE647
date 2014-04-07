#pragma once

/// @fixme Need multiple GLEW context, single GLEW context is prone to error
/// @fixme Need to separate window manager and window

#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include "GLFW/glfw3.h"
#include <string>
#include <iostream>
#include <map>
#include <list>
#include <vector>
#include <assert.h>
using namespace std;

class MouseState {
public:
	enum Button {
		Left = GLFW_MOUSE_BUTTON_LEFT,		
		Right = GLFW_MOUSE_BUTTON_RIGHT,
		Middle = GLFW_MOUSE_BUTTON_MIDDLE
	};

	enum State {
		Up,
		Down
	};

	void setPosition(double x, double y) { xpos = x; ypos = y; }
	void setState(Button btn, State sta) { state[btn] = sta; }
	State getState(Button btn) { return state[btn]; }
	double x() const { return xpos; }
	double y() const { return ypos; }

private:
	State state[3];
	double xpos, ypos;
};

class GLFWWindow {
protected:
	friend class GLFWWindowManager;
	GLFWWindow(){}
	GLFWWindow(int w, int h, const string& title);
	virtual ~GLFWWindow();

public:
	int width() const { return w; }
	int height() const { return h; }
	
	void makeCurrent() const;
	void doneCurrent() const;

	virtual void destroy();
	virtual bool init();
	virtual void display();

	virtual void resize(int width, int height);
	virtual void keyboard(int, int, int, int);
	virtual void mouse(int, int, int);
	virtual void cursor_pos(double, double);

	virtual void screenshot(const string& filename);
	GLFWwindow* getHandle() { return window; }

protected:
	GLFWwindow *window;
	string title;
	int w, h;

	MouseState mouseState;
};

class GLFWWindowManager
{
public:
	static GLFWWindowManager* instance() {
		if( manager == nullptr ) { manager = new GLFWWindowManager; }
		return manager;
	}

	void run();
	GLFWWindow* createWindow(int w, int h, const string& title);
	void registerWindow(GLFWWindow* window);

protected:
	static void keyboard_callback(GLFWwindow*, int, int, int, int);
	static void mouse_callback(GLFWwindow*, int, int, int);
	static void cursorpos_callback(GLFWwindow*, double, double);
	static void resize_callback(GLFWwindow*, int, int);

	static bool shouldClose();


protected:
	static GLFWWindow* getHandle(GLFWwindow*);
	static map<GLFWwindow*, GLFWWindow*> windowmap;
	static GLFWWindowManager* manager;

private:
	// noncopyable
	GLFWWindowManager(){}
	~GLFWWindowManager() { if( manager != nullptr ) delete manager; }

	GLFWWindowManager(const GLFWWindowManager&);
	GLFWWindowManager(const GLFWWindowManager&&);
	GLFWWindowManager& operator=(const GLFWWindowManager&);
	GLFWWindowManager& operator=(const GLFWWindowManager&&);
};


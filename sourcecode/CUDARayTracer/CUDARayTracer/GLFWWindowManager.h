#pragma once

#include "GLFW/glfw3.h"

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

class GLFWWindowManager
{
public:
	GLFWWindowManager(void);
	~GLFWWindowManager(void);
};


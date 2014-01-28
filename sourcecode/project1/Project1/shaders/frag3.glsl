#version 400 core
struct Dummy {
	float a;
	vec3 color;
};

Dummy d[2] = {
	Dummy(1.0, vec3(1, 2, 3)),
	Dummy(2.0, vec3(1, 0, 1))
};

//void init() {
//	d[0] = Dummy(1.0, vec3(1, 2, 3));
//}

out vec4 color;

void main() {
	//init();
	color = vec4(1, 0, 0, 1);
}
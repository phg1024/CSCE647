#version 400
in vec3 vp;

void main(void) {
	gl_Position = vec4(vp, 1.0);
}
/*
void main()
{
    // Transforming The Vertex
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
*/

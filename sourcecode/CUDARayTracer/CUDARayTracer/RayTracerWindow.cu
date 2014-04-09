#include "RayTracerWindow.h"

#include "utils.h"

void CUDARayTracer::createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
	assert(vbo);

	if( !(*vbo) ) {
		cout << "generating new vbo..." << endl;
		// create buffer object
		glGenBuffers(1, vbo);
	}
	else {
		cout << "unregister vbo ..." << endl;
		checkCudaErrors(cudaGraphicsUnregisterResource(*vbo_res));
		cout << "done" << endl;
	}
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = npixels() * 3 * sizeof(float);
	cout << size << endl;
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
	showCUDAMemoryUsage();
}

void CUDARayTracer::deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res) {
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(vbo_res);

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

void CUDARayTracer::loadScene(const string& filename) {
	showCUDAMemoryUsage();

	if(!scene.load("scene0.txt")) cout << "scene file loading failed!" << endl;
	else {
		shapes = scene.getShapes();
		materials = scene.getMaterials();
		lights = scene.getLights();
		cout << "scene loaded. " << endl
			 << shapes.size() << " shapes in total."  << endl
			 << lights.size() << " lights in total."  << endl
			 << materials.size() << " materials in total."  << endl
			 << scene.getTextures().size() << " textures in total." 
			 << endl;
	}

	gamma = scene.gammaValue();
	tracingType = scene.tracingType();

	cam = scene.camera();	

	cudaMalloc((void**)&d_cam, sizeof(Camera));
	cudaMemcpy(d_cam, &cam, sizeof(Camera), cudaMemcpyHostToDevice);

	const size_t sz_shapes = shapes.size() * sizeof(Shape);
	cudaMalloc((void**)&d_shapes, sz_shapes);
	cudaMemcpy(d_shapes, &(shapes[0]), sz_shapes, cudaMemcpyHostToDevice);

	const size_t sz_mats = materials.size() * sizeof(Material);
	cudaMalloc((void**)&d_materials, sz_mats);
	cudaMemcpy(d_materials, &(materials[0]), sz_mats, cudaMemcpyHostToDevice);
		
	const size_t sz_lights = lights.size() * sizeof(int);
	cudaMalloc((void**)&d_lights, sz_lights);
	cudaMemcpy(d_lights, &(lights[0]), sz_lights, cudaMemcpyHostToDevice);

	const vector<TextureObject>& texs = scene.getTextures();
	const size_t sz_tex = sizeof(TextureObject)*texs.size();
	cout << "sz_tex = " << sz_tex << endl;
	cudaMalloc((void**)&d_tex, sz_tex);
	cudaMemcpy(d_tex, &(texs[0]), sz_tex, cudaMemcpyHostToDevice);

	// create texture objects for textures
	vector<cudaTextureObject_t> texobjs;
	for(int i=0;i<texs.size();i++) {
		// Allocate CUDA array in device memory
		cudaChannelFormatDesc channelDesc;
		size_t elemSize;
		cudaTextureReadMode tMode;
		int nMode;

		switch( texs[i].t ) {
		case TextureObject::Mesh:		// v0 v1 v2 mid
		case TextureObject::HDRImage:	// rgba
			nMode = (texs[i].t==TextureObject::Mesh)?0:1;
			channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
			elemSize = sizeof(float4);
			tMode = cudaReadModeElementType;
			break;
		case TextureObject::Normal:	// st
			nMode = 0;
			channelDesc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
			elemSize = sizeof(float3);
			tMode = cudaReadModeElementType;
			break;
		case TextureObject::TextureCoordinates:	// st
			nMode = 0;
			channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
			elemSize = sizeof(float2);
			tMode = cudaReadModeElementType;
			break;
		case TextureObject::Image:
			nMode = 1;
			channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
			elemSize = sizeof(uchar4);
			tMode = cudaReadModeNormalizedFloat;
			break;
		}

		cudaArray* cuArray;
		cudaMallocArray(&cuArray, &channelDesc, texs[i].size.x, texs[i].size.y);

		// Copy texture data to device side
		cudaMemcpyToArray(cuArray, 0, 0, texs[i].addr, texs[i].size.x*texs[i].size.y*elemSize, cudaMemcpyDeviceToDevice);

		// create texture object
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0]   = cudaAddressModeWrap;
		texDesc.addressMode[1]   = cudaAddressModeWrap;
		texDesc.filterMode       = cudaFilterModeLinear;
		texDesc.readMode         = tMode;
		texDesc.normalizedCoords = nMode;

		// create texture object: we only have to do this once!
		cudaTextureObject_t tex=0;
		cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

		texobjs.push_back(tex);
	}
	size_t sz_texobjs = sizeof(cudaTextureObject_t)*texobjs.size();
	cudaMalloc((void**)&d_texobjs, sz_texobjs);
	cudaMemcpy(d_texobjs, &(texobjs[0]), sz_texobjs, cudaMemcpyHostToDevice);
	
	cout << d_texobjs << endl;

	cout << "scene initialized." << endl;
}


RayTracerWindow::RayTracerWindow(int w, int h, const string& title)
{
	this->w = w;
	this->h = h;
	this->title = title;

	init();
}

RayTracerWindow::~RayTracerWindow(void)
{
}

void RayTracerWindow::destroy() {
	GLFWWindow::destroy();
}

bool RayTracerWindow::init() {
	GLFWWindow::init();

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, w, h);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, w, 0, h, 0.1, 10.0);

	tball.init();
	tball.reshape(w, h);
	tball.setSceneScale(1.0);

	return true;
}

void RayTracerWindow::display() {	
	renderer->render();

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -1.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, renderer->vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4,GL_UNSIGNED_BYTE,12,(GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, w * h);
	glDisableClientState(GL_VERTEX_ARRAY);

	//glutSwapBuffers();
	glFlush();	
	cudaThreadSynchronize();

	renderer->computeFPS();

	if( renderer->iterations == renderer->scene.maxIters() ){ 
		screenshot(renderer->scene.sceneName());

		// hard coded
		exit(0);
	}
}

void RayTracerWindow::resize(int w, int h) {
	// make sure the context is correct
	makeCurrent();

	cout << "resizing canvas to " << w << "x" << h << endl;

	// @fixme need to clean up this mess
	glfwSetWindowSize(window, w, h);
	GLFWWindow::resize(w, h);
	tball.reshape(w, h);
	renderer->resize(w, h);

	// viewport
	glViewport(0, 0, w, h);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, w, 0, h, 0.1, 10.0);
}

void RayTracerWindow::keyboard(int key, int scancode, int action, int mods) {
	if( action != GLFW_PRESS ) return;
	switch (key)
	{
	case GLFW_KEY_1:
	case GLFW_KEY_2:
	case GLFW_KEY_3:
		renderer->sMode = key - '0';
		break;
	case GLFW_KEY_G:
		cout << "input gamma value: " << endl;
		cin >> renderer->gamma;
		break;
	case GLFW_KEY_A:
		cout << "Please input number of samples:" << endl;
		cin >> renderer->AASamples;
		break;
	case GLFW_KEY_K:
		renderer->kernelIdx = (renderer->kernelIdx + 1) % 3;
		cout << "using kernel #" << renderer->kernelIdx << endl;
		break;
	case GLFW_KEY_T:
		renderer->tracingType = (renderer->tracingType + 1) % 3;
		cout << "tracing type = " << renderer->tracingType << endl;
		renderer->clear();
		break;
	case GLFW_KEY_S:
		renderer->specType = (renderer->specType + 1) % 5;
		break;
	case GLFW_KEY_C:
		screenshot(renderer->scene.sceneName());
		break;
	case GLFW_KEY_ESCAPE:
		glfwTerminate();
		break;
	}
}

void RayTracerWindow::mouse(int button, int action, int mods) {
	if( action == GLFW_PRESS ) {
		renderer->AASamples_old = renderer->AASamples; renderer->AASamples = 1;
		mouseState.setState(MouseState::Button(button), MouseState::Down);
	}
	else {
		renderer->AASamples = renderer->AASamples_old;
		mouseState.setState(MouseState::Button(button), MouseState::Up);
	}

	double x, y;
	glfwGetCursorPos(window, &x, &y);
	mouseState.setPosition(x, y);

	switch( button ) {
	case GLFW_MOUSE_BUTTON_LEFT:
		if( action == GLFW_PRESS ) {
			tball.mouse_rotate(x, y);
		}
		break;
	case GLFW_MOUSE_BUTTON_RIGHT:
		break;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		break;
	}
	renderer->clear();
}

void RayTracerWindow::cursor_pos(double x, double y) {
	double dx, dy;
	dx = x - mouseState.x();
	dy = y - mouseState.y();

	if( mouseState.getState(MouseState::Left) == MouseState::Down ) {
		tball.motion_rotate(x, y);
		mouseState.setPosition(x, y);
		renderer->clear();
	}
	else if( mouseState.getState(MouseState::Right) == MouseState::Down ) {
		tball.wheel( dy );
		mouseState.setPosition(x, y);
		renderer->clear();
	}
}

void RayTracerWindow::screenshot(const string& filename) {
	GLFWWindow::screenshot(filename + ".png");

	// save HDR image
	vector<float3> buffer(w*h);
	vector<float3> flipped(w*h);
	cudaMemcpy(&buffer[0], renderer->cumulatedColor, sizeof(float3)*w*h, cudaMemcpyDeviceToHost);
	for(int i=0,idx=0;i<h;++i) {
		int offset = (h-1-i)*w;
		for(int j=0;j<w;++j,++idx,++offset) {
			flipped[offset] = buffer[idx];
		}
	}

	string hdrfilename = filename + ".hdr";
	FILE* f = fopen(hdrfilename.c_str(),"wb");
	RGBE_WriteHeader(f,w,h,NULL);
	RGBE_WritePixels(f,&(flipped[0].x),w*h);
	fclose(f);
}
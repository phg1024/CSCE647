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

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include <cuda_texture_types.h>

#include <vector_types.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "element.h"
#include "definitions.h"
#include "Scene.h"
#include "trackball.h"

#include "FreeImagePlus.h"

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD         0.30f
#define REFRESH_DELAY     25 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
unsigned int edgeX = 8, edgeY = 8;
unsigned int window_width  = 1024 + edgeX;
unsigned int window_height = 768 + edgeY;

// vbo variables
GLuint vbo = 0;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;
TrackBall tball;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
			   unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
extern __global__ void setParams(int, int, int);
extern __global__ void bindTexture2(const cudaTextureObject_t* texs, int texCount);

extern __global__ void copy2pbo(float3*, float3*, int, int, int, float);
extern __global__ void clearCumulatedColor(float3*, int, int);

extern __global__ void raytrace(float time, float3 *pos, Camera* cam, 
						 int nLights, int* lights, 
						 int nShapes, Shape* shapes,
						 int nMaterials, Material* materials,
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples);

extern __global__ void raytrace2(float time, float3 *pos, Camera* cam, 
						 int nLights, int* lights, 
						 int nShapes, Shape* shapes, 
						 int nMaterials, Material* materials,
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples, 
						 int gx, int gy, int gmx, int gmy);

extern __global__ void initCurrentBlock(int v);

extern __global__ void raytrace3(float time, float3 *pos, Camera* cam, 
						 int nLights, int* lights, 
						 int nShapes, Shape* shapes, 
						 int nMaterials, Material* materials,
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples, 
						 int bmx, int bmy, int tlb);

void runCuda(struct cudaGraphicsResource **vbo_resource);


Scene scene;

Camera cam;
Camera* d_cam;
vector<Shape> shapes;
Shape* d_shapes;
TextureObject* d_tex;
cudaTextureObject_t* d_texobjs;
vector<int> lights;
int* d_lights;
vector<Material> materials;
Material* d_materials;
float3* cumulatedColor = 0;
int AASamples = 1;
int sMode = 1;
int kernelIdx = 0;
int specType = 0;
int tracingType = 0;
int iterations = 0;
float gamma = 1.0;

void init_scene()
{
	showCUDAMemoryUsage();
	const size_t sz = sizeof(Camera);
	cudaMalloc((void**)&d_cam, sz);
	cudaMemcpy(d_cam, &cam, sz, cudaMemcpyHostToDevice);

	if(!scene.load("scene0.txt")) cout << "scene file loading failed!" << endl;
	else {
		shapes = scene.getShapes();
		materials = scene.getMaterials();
		lights = scene.getLights();
		cout << "scene loaded. " 
			 << shapes.size() << " shapes in total." 
			 << materials.size() << " materials in total." 
			 << scene.getTextures().size() << " textures in total." 
			 << endl;
	}

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
		if( texs[i].isHDR )
			channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		else
			channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaArray* cuArray;
		cudaMallocArray(&cuArray, &channelDesc, texs[i].size.x, texs[i].size.y);

		// Copy to device memory some data located at address h_data
		// in host memory
		if( texs[i].isHDR )
			cudaMemcpyToArray(cuArray, 0, 0, texs[i].addr, texs[i].size.x*texs[i].size.y*sizeof(float4), cudaMemcpyDeviceToDevice);
		else
			cudaMemcpyToArray(cuArray, 0, 0, texs[i].addr, texs[i].size.x*texs[i].size.y*sizeof(uchar4), cudaMemcpyDeviceToDevice);

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
		if( texs[i].isHDR )
			texDesc.readMode         = cudaReadModeElementType;
		else
			texDesc.readMode         = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		// create texture object: we only have to do this once!
		cudaTextureObject_t tex=0;
		cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

		texobjs.push_back(tex);
	}
	size_t sz_texobjs = sizeof(cudaTextureObject_t)*texobjs.size();
	cudaMalloc((void**)&d_texobjs, sz_texobjs);
	cudaMemcpy(d_texobjs, &(texobjs[0]), sz_texobjs, cudaMemcpyHostToDevice);

	cout << "scene initialized." << endl;
}

void launch_kernel(float3 *pos, unsigned int mesh_width,
				   unsigned int mesh_height, int sMode)
{
	// update camera info	
	mat4 mat(tball.getInverseMatrix());
	mat = mat.trans();

	vec3 camPos = cam.pos;
	vec3 camDir = cam.dir;
	vec3 camUp = cam.up;

	camPos = (mat * (camPos / tball.getScale()));
	camDir = (mat * camDir);
	camUp = (mat * camUp);

	Camera caminfo = cam;
	caminfo.dir = camDir;
	caminfo.up = camUp;
	caminfo.pos = camPos;
	caminfo.right = caminfo.dir.cross(caminfo.up);
	
	cudaMemcpyAsync(d_cam, &caminfo, sizeof(Camera), cudaMemcpyHostToDevice);

	bindTexture2<<< 1, 1 >>>(d_texobjs, scene.getTextures().size());
	setParams<<<1, 1>>>(specType, tracingType, scene.getEnvironmentMap());

	switch( kernelIdx ) {
	case 0:{
		// execute the kernel
		dim3 block(32, 32, 1);
		dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
		raytrace<<< grid, block >>>((iterations+rand()%1024), cumulatedColor, d_cam,
			lights.size(), d_lights,
			shapes.size(), d_shapes,
			materials.size(), d_materials,
			window_width, window_height, sMode, AASamples);
		break;
		   }
	case 1:{
		dim3 block(32, 32, 1);
		dim3 group(4, 4, 1);
		dim3 grid(group.x, group.y, 1);
		dim3 groupCount(ceil(window_width/(float)(block.x * group.x)), ceil(window_height/(float)(block.y * group.y)), 1);

		raytrace2<<< grid, block >>>((iterations+rand()%1024), cumulatedColor, d_cam,
			lights.size(), d_lights,
			shapes.size(), d_shapes,
			materials.size(), d_materials,
			window_width, window_height, sMode, AASamples, 
			group.x, group.y, groupCount.x, groupCount.y);
		break;
		   }
	case 2:{
		dim3 block(32, 32, 1);
		dim3 grid(4, 4, 1);

		dim3 blockCount(ceil(window_width/(float)block.x), ceil(window_height/(float)block.y ), 1);

		unsigned totalBlocks = blockCount.x*blockCount.y;
		//cout << "total blocks = " << totalBlocks << endl;
		srand(clock());

		initCurrentBlock<<<1, 1>>>(0);
		raytrace3<<< grid, block >>>((iterations+rand()%1024), cumulatedColor, d_cam,
			lights.size(), d_lights,
			shapes.size(), d_shapes,
			materials.size(), d_materials,
			window_width, window_height, sMode, AASamples, 
			blockCount.x, blockCount.y, totalBlocks);
		break;
		   }
	}
	cudaThreadSynchronize();

	iterations++;
	//cout << iterations << endl;

	// copy to pbo
	dim3 block(32, 32, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	copy2pbo<<<grid,block>>>(cumulatedColor, pos, iterations, window_width, window_height, gamma);
	cudaThreadSynchronize();
}

bool checkHW(char *name, const char *gpuType, int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy(name, deviceProp.name);

	if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findGraphicsGPU(char *name)
{
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[256], temp[256];

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("> FAILED program finished, exiting...\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
		printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

		if (bGraphics)
		{
			if (!bFoundGraphics)
			{
				strcpy(firstGraphicsName, temp);
			}

			nGraphicsGPU++;
		}
	}

	if (nGraphicsGPU)
	{
		strcpy(name, firstGraphicsName);
	}
	else
	{
		strcpy(name, "this hardware");
	}

	return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	FreeImage_Initialise();

	srand(clock());
	
	char *ref_file = NULL;

	pArgc = &argc;
	pArgv = argv;

	//printf("%s starting...\n");

	if (argc > 1)
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "file"))
		{
			// In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
			getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
		}
	}

	printf("\n");

	runTest(argc, argv, ref_file);

	FreeImage_DeInitialise();
	return 0;
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "CUDA Ray Tracer: %3.1f fps - Iteration %d", avgFPS, iterations);
	glutSetWindowTitle(fps);
}


void resize(int w, int h) 
{
	tball.reshape(w, h);
	//return;
	cout << w << "x" << h << " vs " << window_width << "x" << window_height << endl;

	if( w == window_width &&  h == window_height ) return;

	window_width = w, window_height = h;
	// camera
	cam.h = h / (float) w;

	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	showCUDAMemoryUsage();

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, window_width, 0, window_height, 0.1, 10.0);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE );
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("CUDA Ray Tracer");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	//glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
	glewInit();

	if (! glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, window_width, 0, window_height, 0.1, 10.0);

	tball.init();
	tball.setSceneScale(1.0);

	SDK_CHECK_ERROR_GL();

	return true;
}

void refresh() {
	//system("pause");
	glutPostRedisplay();
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
		{
			return false;
		}
	}
	else
	{
		cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	}

	
	//size_t nStack;
	//cudaDeviceGetLimit(&nStack, cudaLimitStackSize);
	//cout << "stack size = " << nStack << endl;
	//cudaDeviceSetLimit(cudaLimitStackSize, 65536);
	//cudaDeviceGetLimit(&nStack, cudaLimitStackSize);
	//cout << "stack size = " << nStack << endl;
	

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	// initialize the scene on CUDA kernels
	init_scene();
	
	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);	

	glutReshapeFunc(resize);
	glutIdleFunc(refresh);

	// run the cuda part
	runCuda(&cuda_vbo_resource);

	// start rendering mainloop
	glutMainLoop();
	atexit(cleanup);

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float3 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	launch_kernel(dptr, window_width, window_height, sMode);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
	printf("sdkDumpBin: <%s>\n", filename);
	FILE *fp;
	FOPEN(fp, filename, "wb");
	fwrite(data, bytes, 1, fp);
	fflush(fp);
	fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
			   unsigned int vbo_res_flags)
{
	assert(vbo);

	if( !(*vbo) ) {
		cout << "generating new vbo..." << endl;
		// create buffer object
		glGenBuffers(1, vbo);
	}
	else {
		cout << "unregister vbo ..." << endl;
		cudaGraphicsUnregisterResource(*vbo_res);
	}
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = window_width * window_height * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();

	// allocate memory
	if( cumulatedColor ) cudaFree(cumulatedColor);
	int sz = window_width * window_height * sizeof(float3);
	cudaMalloc((void**)&cumulatedColor, sz);
	cudaMemset(cumulatedColor, 0, sz);
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(vbo_res);

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -1.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4,GL_UNSIGNED_BYTE,12,(GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, window_width * window_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	//glutSwapBuffers();
	glFlush();

	sdkStopTimer(&timer);
	computeFPS();
}

void clearColor() {
	dim3 block(32, 32, 1);
	dim3 grid(window_width / block.x, window_height / block.y, 1);
	clearCumulatedColor<<<grid,block>>>(cumulatedColor, window_width, window_height);
	iterations = 0;
	cudaThreadSynchronize();
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}

	cudaFree(d_cam);
	cudaFree(d_shapes);
	cudaFree(d_lights);

	cudaDeviceReset();
	printf("program completed, returned %s\n", (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void screenshot() {
	// Make the BYTE array, factor of 3 because it's RBG.
	int width = window_width - edgeX;
	int height = window_height - edgeX;
	BYTE* pixels = new BYTE[ 3 * width * height];
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

	cv::Mat m( height, width, CV_8UC3, pixels );
	cv::cvtColor(m, m, CV_RGB2BGR);
	cv::flip(m, m, 0);
	cv::imwrite("screenshot.png", m);
	delete[] pixels;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case '1':
	case '2':
	case '3':
		sMode = key - '0';
		glutPostRedisplay();
		break;
	case 'g':
	case 'G':
		cout << "input gamma value: " << endl;
		cin >> gamma;
		glutPostRedisplay();
		break;
	case 'a':
	case 'A':
		cout << "Please input number of samples:" << endl;
		cin >> AASamples;
		glutPostRedisplay();
		break;
	case 'k':
	case 'K':
		kernelIdx = (kernelIdx + 1) % 3;
		cout << "using kernel #" << kernelIdx << endl;
		glutPostRedisplay();
		break;
	case 't':
	case 'T':
		tracingType = (tracingType + 1) % 3;
		cout << "tracing type = " << tracingType << endl;
		clearColor();
		glutPostRedisplay();
		break;
	case 's':
	case 'S':
		specType = (specType + 1) % 5;
		glutPostRedisplay();
		break;
	case 'c':
	case 'C':
		screenshot();
		break;
	case (27) :
		cleanup();
		glutLeaveMainLoop();
		break;
	}
}

int AASamples_old;
////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1<<button;
		AASamples_old = AASamples;
		AASamples = 1;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
		AASamples = AASamples_old;
	}

	if (mouse_buttons & 1)
	{
		tball.mouse_rotate(x, y);
	}
	else if (mouse_buttons & 4)
	{		
	}

	clearColor();

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		tball.motion_rotate(x, y);
	}
	else if (mouse_buttons & 4)
	{
		tball.wheel( y - mouse_old_y );
	}

	mouse_old_x = x;
	mouse_old_y = y;

	clearColor();
	glutPostRedisplay();
}

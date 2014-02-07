// includes, system
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

#include "FreeImage.h"

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

#include "element.h"
#include "definitions.h"
#include "trackball.h"

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
extern __global__ void setParams(int, int);
extern __global__ void bindTexture(uchar4**, int2* );

extern __global__ void initScene(int nLights, Light* lights, 
								 int nShapes, Shape* shapes);

extern __global__ void raytrace(float3 *pos, Camera* cam, 
						 int nLights, Light* lights, 
						 int nShapes, Shape* shapes, 
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples);

extern __global__ void raytrace2(float3 *pos, Camera* cam, 
						 int nLights, Light* lights, 
						 int nShapes, Shape* shapes, 
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples, 
						 int gx, int gy, int gmx, int gmy);

extern __global__ void initCurrentBlock(int v);

extern __global__ void raytrace3(float3 *pos, Camera* cam, 
						 int nLights, Light* lights, 
						 int nShapes, Shape* shapes, 
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples, 
						 int bmx, int bmy, int tlb);

void runCuda(struct cudaGraphicsResource **vbo_resource);

Camera cam;
Camera* d_cam;
thrust::host_vector<Shape> shapes;
Shape* d_shapes;
int textureCount = 0;
float* texAddr[32];
uchar4** d_texAddr;
int2 texSize[32];
int2* d_texSize;
//texture<float, cudaTextureType2D, cudaReadModeElementType> texObjs[32];
thrust::host_vector<Light> lights;
Light* d_lights;
int AASamples = 4;
int sMode = 2;
int kernelIdx = 0;
int specType = 0;
int tracingType = 0;

int loadTexture(const char* filename) {
	cv::Mat image = cv::imread(filename); 

	cv::imshow("texture", image);

	cout << image.cols << "x" << image.rows << endl;
	int width = image.cols, height = image.rows;
	vector<uchar4> buffer(width*height*3);
	for(int i=0;i<height;i++) {
		for(int j=0;j<width;j++) {
			cv::Vec3b bgrPixel = image.at<cv::Vec3b>(i, j);
			int idx = (i*width+j);
			buffer[idx] = make_uchar4(bgrPixel.val[2], bgrPixel.val[1], bgrPixel.val[0], 255);
		}
	}

	texSize[textureCount] = make_int2(width, height);
	const size_t sz_tex = width * height * sizeof(uchar4);
	cudaMalloc((void**)&(texAddr[textureCount]), sz_tex);
	cudaMemcpy(texAddr[textureCount], &buffer[0], sz_tex, cudaMemcpyHostToDevice);

	return textureCount++;
}

void init_scene()
{
	// initialize the camera
	cam.pos = vec3(1, 1, -5);
	cam.dir = vec3(-0.5, -1.5, 5).normalized();
	cam.up = vec3(0, 5, 1.5).normalized();
	cam.right = vec3(1, 0, 0);
	cam.f = 1.0;
	cam.w = 1.0; cam.h = window_height / (float) window_width; 

	const size_t sz = sizeof(Camera);
	cudaMalloc((void**)&d_cam, sz);
	cudaMemcpy(d_cam, &cam, sz, cudaMemcpyHostToDevice);

	/*
	int scount = 7;
	// initialize the scene by uploading scene objects to GPU
	for(int i=0;i<scount;i++)
		for(int j=0;j<scount;j++)
		{
			vec3 pos(i*4-scount*2, 0.0, j*4-scount*2);
			shapes.push_back(
				Shape::createSphere(pos, 1.0, Material(
				random3(i*13+j),		// diffuse
				vec3(1.0, 1.0, 1.0),		// specular
				vec3(0.10, 0.10, 0.1),		// ambient
				50.0f,							// shininess
				vec3(0, 0, .4),				// kcool
				vec3(.4, .4, 0),				// kwarm
				0.15, 0.25,
				0.75, 0.25, 0.0))
				);
		}
		*/

	// make a box
	// left
	shapes.push_back(Shape::createPlane(
		vec3(-3, 3, 1),
		4.0, 4.0,
		vec3(1, 0, 0),
		vec3(0, 0, 1),
		vec3(0, 1, 0),
		Material(
		vec3(0.95, 0.25, 0.25),
		vec3(1, 1, 1),
		vec3(0.05, 0.05, 0.05),
		50.0,
		vec3(0, 0, .4),
		vec3(.4, .4, 0),
		0.15, 0.25,
		0.95, 0.05, 0.0
		)
		));
	// right
	shapes.push_back(Shape::createPlane(
		vec3(3, 3, 1),
		4.0, 4.0,
		vec3(-1, 0, 0),
		vec3(0, 0, 1),
		vec3(0, 1, 0),
		Material(
		vec3(0.25, 0.95, 0.25),
		vec3(1, 1, 1),
		vec3(0.05, 0.05, 0.05),
		50.0,
		vec3(0, 0, .4),
		vec3(.4, .4, 0),
		0.15, 0.25,
		0.95, 0.05, 0.0
		)
		));
	// back
	shapes.push_back(Shape::createPlane(
		vec3(0, 3, 5),
		4.0, 4.0,
		vec3(0, 0, -1),
		vec3(1, 0, 0),
		vec3(0, 1, 0),
		Material(
		vec3(0.75, 0.75, 0.75),
		vec3(1, 1, 1),
		vec3(0.05, 0.05, 0.05),
		50.0,
		vec3(0, 0, .4),
		vec3(.4, .4, 0),
		0.15, 0.25,
		0.95, 0.05, 0.0
		)
		));

	Shape sp = Shape(Shape::PLANE,
		vec3(0, -1, 0),
		500.0, 500.0, 0.0,
		vec3(0, 1, 0),
		vec3(1, 0, 0),
		vec3(0, 0, 1),
		Material(
		vec3(0.75, 0.75, 0.75),
		vec3(1, 1, 1),
		vec3(0.05, 0.05, 0.05),
		50.0,
		vec3(0, 0, .4),
		vec3(.4, .4, 0),
		0.15, 0.25,
		0.75, 0.25, 0.0));
	sp.hasTexture = true;
	sp.texId = loadTexture("./textures/chessboard.png");
	//sp.texId = loadTexture("./textures/wood-texture.jpg");
	cout << "tex id = " << sp.texId << endl;
	shapes.push_back( sp );
	
	shapes.push_back( Shape::createSphere(
		vec3(0, -0.5, -0.5), 0.5, 
		Material(
		vec3(0.95, 0.25 , 0.0 ),		// diffuse
		vec3(1.0 , 1.0 , 1.0 ),		// specular
		vec3(0.05, 0.10, 0.15),		// ambient
		50.0f,							// shininess
		vec3(0, 0, .4),				// kcool
		vec3(.4, .4, 0),				// kwarm
		0.15, 0.25,
		0.15, 0.05, 0.8))
		);
	shapes.push_back( Shape::createSphere(
		vec3(0.5, -0.75, -1),	// p 
		0.25,
		Material(
		vec3(0.25, 0.25, 0.95),		// diffuse
		vec3(1.0 , 1.0 , 1.0),		// specular
		vec3(0.05, 0.05, 0.05),		// ambient
		20.0f,							// shininess
		vec3(0, .4, 0),				// kcool
		vec3(.4, 0, .4),				// kwarm
		0.15, 0.25,
		0.5, 0.3, 0.2))
		);
	shapes.push_back(Shape( Shape::ELLIPSOID, 
		vec3(1.25, -0.5, -0.5),	// p 
		0.5, 0.45, 0.35,		// radius
		vec3(1, 0, 1),			// axis[0]
		vec3(1, 1, 0),			// axis[1]
		vec3(0, 1, 1),			// axis[2]
		Material(
		vec3(0.75, 0.75, 0.25),		// diffuse
		vec3(1.0 , 1.0 , 1.0),		// specular
		vec3(0.05, 0.05, 0.05),		// ambient
		100.0f,							// shininess
		vec3(.2, .9, .6),				// kcool
		vec3(.05, .35, .95),				// kwarm
		0.25, 0.25,
		0.05, 0.95, 0.0
		))
		);
	
	shapes.push_back(Shape( Shape::CYLINDER, 
		vec3(0.5, -1.0, 0.75),	// p 
		0.5, 0.25, 0.25,		// radius
		vec3(0, 1, 0),			// axis[0]
		vec3(1, 0, 0),			// axis[1]
		vec3(0, 0, 1),			// axis[2]
		Material(
		vec3(0.35, 0.95, 0.25),		// diffuse
		vec3(1.0 , 1.0 , 1.0),		// specular
		vec3(0.05, 0.05, 0.05),		// ambient
		100.0f,							// shininess
		vec3(.9, .1, .6),				// kcool
		vec3(.05, .45, .05),				// kwarm
		0.25, 0.15,
		0.5, 0.5, 0.0
		))
		);
	
	
	shapes.push_back(Shape( Shape::CONE, 
		vec3(0.4, -1.0, -1.25),	// p 
		0.25, 0.05, 0.05,		// radius
		vec3(-1.0, atan(0.3), 0),			// axis[0]
		vec3(atan(0.3), 1.0, 0),			// axis[1]
		vec3(0, 0, 1),			// axis[2]
		Material(
		vec3(0.75, 0.75, 0.25),		// diffuse
		vec3(1.0 , 1.0 , 1.0),		// specular
		vec3(0.075, 0.05, 0.05),		// ambient
		100.0f,							// shininess
		vec3(.9, .1, .6),				// kcool
		vec3(.05, .45, .05),				// kwarm
		0.25, 0.15,
		0.15, 0.1, 0.75
		))
		);

	
	shapes.push_back(Shape( Shape::HYPERBOLOID, 
		vec3(-1.5, 0.25, 2.0),	// p 
		0.2, 0.1, 0.1,		// radius
		vec3(0, 1, 0),			// axis[0]
		vec3(1, 0, 0),			// axis[1]
		vec3(0, 0, 1),			// axis[2]
		Material(
		vec3(0.75, 0.75, 0.25),		// diffuse
		vec3(1.0 , 1.0 , 1.0),		// specular
		vec3(0.075, 0.05, 0.05),		// ambient
		100.0f,							// shininess
		vec3(.9, .1, .6),				// kcool
		vec3(.05, .45, .05),				// kwarm
		0.25, 0.15,
		0.8, 0.2, 0.0
		))
		);

	shapes.push_back(Shape( Shape::HYPERBOLOID2, 
		vec3(2.5, 0.25, 2.0),	// p 
		0.125, 0.1, 0.1,		// radius
		vec3(0, 1, 0),			// axis[0]
		vec3(1, 0, 0),			// axis[1]
		vec3(0, 0, 1),			// axis[2]
		Material(
		vec3(0.25, 0.75, 0.75),		// diffuse
		vec3(1.0 , 1.0 , 1.0),		// specular
		vec3(0.075, 0.05, 0.05),		// ambient
		100.0f,							// shininess
		vec3(.9, .1, .6),				// kcool
		vec3(.05, .45, .05),				// kwarm
		0.25, 0.15,
		0.1, 0.1, 0.8
		))
		);

	const size_t sz_shapes = shapes.size() * sizeof(Shape);
	cudaMalloc((void**)&d_shapes, sz_shapes);
	cudaMemcpy(d_shapes, &(shapes[0]), sz_shapes, cudaMemcpyHostToDevice);

	lights.push_back(Light(Light::DISK, 0.5, vec3(1, 1, 1), vec3(1, 1, 1), vec3(1, 1, 1), vec3(-2, 4, -10), vec3(2, -4, 10)));
	lights.push_back(Light(Light::DISK, 0.25, vec3(1, 1, 1), vec3(1, 1, 1), vec3(1, 1, 1), vec3(4, 4, -10)));
	lights.push_back(Light(Light::DISK, 0.1, vec3(1, 1, 1), vec3(1, 1, 1), vec3(1, 1, 1), vec3(1000, 1000, -1000)));
	lights.push_back(Light(Light::DIRECTIONAL, 0.15, vec3(1, 1, 1), vec3(1, 1, 1), vec3(1, 1, 1), vec3(0, 10, -10), vec3(0, -10, 10)));

	const size_t sz_lights = lights.size() * sizeof(Light);
	cudaMalloc((void**)&d_lights, sz_lights);
	cudaMemcpy(d_lights, &(lights[0]), sz_lights, cudaMemcpyHostToDevice);

	const size_t sz_texAddr = sizeof(uchar4*)*32;
	cudaMalloc((void**)&d_texAddr, sz_texAddr);
	cudaMemcpy(d_texAddr, &(texAddr[0]), sz_texAddr, cudaMemcpyHostToDevice);

	const size_t sz_texSize = sizeof(int2)*32;
	cudaMalloc((void**)&d_texSize, sz_texSize);
	cudaMemcpy(d_texSize, &(texSize[0]), sz_texSize, cudaMemcpyHostToDevice);

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
	caminfo.right = -caminfo.dir.cross(caminfo.up);
	
	cudaMemcpyAsync(d_cam, &caminfo, sizeof(Camera), cudaMemcpyHostToDevice);

	bindTexture<<< 1, 1 >>>(d_texAddr, d_texSize);
	setParams<<<1, 1>>>(specType, tracingType);
	/*
	initScene<<<1, dim3(8,8,1)>>>(lights.size(), thrust::raw_pointer_cast(&d_lights[0]),
								  shapes.size(), thrust::raw_pointer_cast(&d_shapes[0]));
	*/

	switch( kernelIdx ) {
	case 0:{
		// execute the kernel
		dim3 block(32, 32, 1);
		dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
		raytrace<<< grid, block >>>(pos, d_cam,
			lights.size(), thrust::raw_pointer_cast(&d_lights[0]),
			shapes.size(), thrust::raw_pointer_cast(&d_shapes[0]), 
			window_width, window_height, sMode, AASamples);
		break;
		   }
	case 1:{
		dim3 block(32, 32, 1);
		dim3 group(4, 4, 1);
		dim3 grid(group.x, group.y, 1);
		dim3 groupCount(ceil(window_width/(float)(block.x * group.x)), ceil(window_height/(float)(block.y * group.y)), 1);

		raytrace2<<< grid, block >>>(pos, d_cam,
			lights.size(), thrust::raw_pointer_cast(&d_lights[0]),
			shapes.size(), thrust::raw_pointer_cast(&d_shapes[0]),
			window_width, window_height, sMode, AASamples, 
			group.x, group.y, groupCount.x, groupCount.y);
		break;
		   }
	case 2:{
		dim3 block(32, 32, 1);
		dim3 grid(2, 2, 1);

		dim3 blockCount(ceil(window_width/(float)block.x), ceil(window_height/(float)block.y ), 1);

		unsigned totalBlocks = blockCount.x*blockCount.y;
		//cout << "total blocks = " << totalBlocks << endl;

		initCurrentBlock<<<1, 1>>>(0);
		raytrace3<<< grid, block >>>(pos, d_cam,
			lights.size(), thrust::raw_pointer_cast(&d_lights[0]),
			shapes.size(), thrust::raw_pointer_cast(&d_shapes[0]), 
			window_width, window_height, sMode, AASamples, 
			blockCount.x, blockCount.y, totalBlocks);
		break;
		   }
	}
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
	char *ref_file = NULL;

	pArgc = &argc;
	pArgv = argv;

	printf("%s starting...\n");

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
	sprintf(fps, "CUDA Ray Tracer: %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}


void showCUDAMemoryUsage() {
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		exit(1);
    }
	
	double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

void resize(int w, int h) 
{
	tball.reshape(w, h);

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

	
	size_t nStack;
	cudaDeviceGetLimit(&nStack, cudaLimitStackSize);
	cout << "stack size = " << nStack << endl;
	cudaDeviceSetLimit(cudaLimitStackSize, 65536);
	cudaDeviceGetLimit(&nStack, cudaLimitStackSize);
	cout << "stack size = " << nStack << endl;
	

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

void timerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
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
	case 's':
	case 'S':
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
		tracingType = (tracingType + 1) % 4;
		cout << "tracing type = " << tracingType << endl;
		glutPostRedisplay();
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
		tball.mouse_rotate(window_width - x, window_height-y);
	}
	else if (mouse_buttons & 4)
	{		
	}

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
		tball.motion_rotate(window_width - x, window_height-y);
	}
	else if (mouse_buttons & 4)
	{
		tball.wheel( y - mouse_old_y );
	}

	mouse_old_x = x;
	mouse_old_y = y;

	glutPostRedisplay();
}

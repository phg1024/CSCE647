#pragma once
#include "GLFWWindowManager.h"

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include <cuda_texture_types.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <vector_types.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/remove.h>

#include "element.h"
#include "definitions.h"
#include "Scene.h"
#include "pixel.h"
#include "trackball.h"

#include "extras/aabbtree/aabbtree.h"

#include "FreeImagePlus.h"

#include "extras/hdr/rgbe.h"

extern __global__ void setParams(int, int, int);
extern __global__ void bindTexture2(const cudaTextureObject_t* texs, int texCount);

extern __global__ void copy2pbo(float3*, float3*, int, int, int, float);
extern __global__ void clearCumulatedColor(float3*, int, int);

extern __global__ void initPixel(PixelState* pixels, int* activeIdx, Camera* cam, int time, int width, int height);
extern __global__ void initScene( int nLights, int* lights, int nShapes, Shape* shapes, int nMaterials, Material* materials);
extern __global__ void raytrace_singlepass(int seed, int pass, PixelState *pixels, int *activeIdx, int startIdx, int endIdx, int batchSize, unsigned int width, unsigned int height);
extern __global__ void collectPixelValue(float3 *color, PixelState* pixels, int width, int height);

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

struct CUDARayTracer;

class RayTracerWindow :	public GLFWWindow
{
public:
	RayTracerWindow(int w, int h, const string& title);
	virtual ~RayTracerWindow(void);

	void bindRenderer(CUDARayTracer* rt) {
		renderer = rt;
	}

	TrackBall& trackball() { return tball; }
	virtual void screenshot(const string& filename);

	virtual bool init();
	virtual void resize(int width, int height);
	virtual void display();

protected:
	virtual void destroy();
	virtual void keyboard(int, int, int, int);
	virtual void mouse(int, int, int);
	virtual void cursor_pos(double, double);

private:
	CUDARayTracer *renderer;

	// interaction
	TrackBall tball;
};


struct CUDARayTracer {
	CUDARayTracer():cumulatedColor(nullptr), vbo(0){
		kernelIdx = 0;
		AASamples = 1;
		sMode = 1;
		specType = 0;
		tracingType = 1;
		iterations = 0;
		gamma = 1.0;

		timer = nullptr;
		fpsCount = 0;        // FPS count for averaging
		fpsLimit = 1;        // FPS limit for sampling
		g_Index = 0;
		avgFPS = 0.0f;
		frameCount = 0;

		updateCamera = false;
	}

	void init() {
		// initialize CUDA GPU
		//cudaDeviceReset();

		sdkCreateTimer(&timer);

		// create window
		window = new RayTracerWindow(640, 480, "CUDA Ray Tracer");
		window->bindRenderer(this);
		GLFWWindowManager::instance()->registerWindow(window);
		SDK_CHECK_ERROR_GL();

		int gpuIdx = gpuGetMaxGflopsDeviceId();
		cout << "using GPU " << gpuIdx << endl;
		cudaGLSetGLDevice(gpuIdx);

		cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, gpuIdx);
        printDevProp(devProp);

		//cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

		// load the scene first
		loadScene("scene0.txt");

		window->resize(scene.width(), scene.height());

		cout << "initializing renderer ..." << endl;
		imagesize.x = scene.width();
		imagesize.y = scene.height();
		createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
		
		int sz = npixels() * sizeof(float3);
		cudaMalloc((void**)&cumulatedColor, sz);
		cudaMemset(cumulatedColor, 0, sz);

		pixels.resize(npixels());
		activePixels0.resize(npixels());
		activePixels1.resize(npixels());

		// set up global settings
		bindTexture2<<< 1, 1 >>>(d_texobjs, scene.getTextures().size());
		setParams<<<1, 1>>>(specType, tracingType, scene.getEnvironmentMap());
		//checkCudaErrors(cudaThreadSynchronize());

		clear();
	}

	void resize(int w, int h) {
		// no need to resize
		if( w == imagesize.x && h == imagesize.y ) return;
		imagesize.x = w;
		imagesize.y = h;
		
		createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
		
		if( cumulatedColor ) cudaFree(cumulatedColor);
		int sz = npixels() * sizeof(float3);
		cudaMalloc((void**)&cumulatedColor, sz);
		cudaMemset(cumulatedColor, 0, sz);

		pixels.resize(npixels());
		activePixels0.resize(npixels());
		activePixels1.resize(npixels());

		cam.h = atan(0.5 * cam.fov / 180.0 * MathUtils::PI) * cam.f;
		cam.w = w / (float)h * cam.h;

		cout << "film size = " << cam.w << "x" << cam.h << endl;
	}

	void computeFPS() {
		frameCount++;
		fpsCount++;

		if (fpsCount == fpsLimit)
		{
			avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
			fpsCount = 0;
			fpsLimit = (int)MAX(avgFPS, 1.f);

			sdkResetTimer(&timer);
		}

		endTime = clock();
		float totalTime = (endTime - startTime)/(float)CLOCKS_PER_SEC;
		char fps[256];
		sprintf(fps, "CUDA Ray Tracer: %3.4f fps - Iteration %d - Elapsed time %3.2f s.", avgFPS, iterations, totalTime);

		glfwSetWindowTitle(window->getHandle(), fps);
	}

	void clear() {
		// need to update camera
		updateCamera = true;

		dim3 block(32, 32, 1);
		dim3 grid(ceil(imagesize.x / (float)block.x), ceil(imagesize.y / (float)block.y), 1);
		clearCumulatedColor<<<grid,block>>>(cumulatedColor, imagesize.x, imagesize.y);
		iterations = 0;
		cudaThreadSynchronize();
		startTime = clock();
	}

	void launch_rendering_kernel(float3 *pos, int sMode) {
		if( updateCamera ) {
			TrackBall& tball = window->trackball();
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

			cudaMemcpy(d_cam, &caminfo, sizeof(Camera), cudaMemcpyHostToDevice);
			updateCamera = false;
		}

		switch( kernelIdx ) {
		case 0:{
			// execute the kernel
			dim3 block(32, 32, 1);
			dim3 grid(ceil(imagesize.x / (float)block.x), ceil(imagesize.y / (float)block.y), 1);
			raytrace<<< grid, block >>>((iterations+rand()), cumulatedColor, d_cam,
				lights.size(), d_lights,
				shapes.size(), d_shapes,
				materials.size(), d_materials,
				imagesize.x, imagesize.y, sMode, AASamples);
			break;
			   }
		case 1:{
			dim3 block(32, 32, 1);
			dim3 group(4, 4, 1);
			dim3 grid(group.x, group.y, 1);
			dim3 groupCount(ceil(imagesize.x/(float)(block.x * group.x)), ceil(imagesize.y/(float)(block.y * group.y)), 1);

			raytrace2<<< grid, block >>>((iterations+rand()), cumulatedColor, d_cam,
				lights.size(), d_lights,
				shapes.size(), d_shapes,
				materials.size(), d_materials,
				imagesize.x, imagesize.y, sMode, AASamples, 
				group.x, group.y, groupCount.x, groupCount.y);
			break;
			   }
		case 2:{
			dim3 block(32, 32, 1);
			dim3 grid(4, 4, 1);

			dim3 blockCount(ceil(imagesize.x/(float)block.x), ceil(imagesize.y/(float)block.y ), 1);

			unsigned totalBlocks = blockCount.x*blockCount.y;
			//cout << "total blocks = " << totalBlocks << endl;
			srand(clock());

			initCurrentBlock<<<1, 1>>>(0);
			raytrace3<<< grid, block >>>((iterations+rand()), cumulatedColor, d_cam,
				lights.size(), d_lights,
				shapes.size(), d_shapes,
				materials.size(), d_materials,
				imagesize.x, imagesize.y, sMode, AASamples, 
				blockCount.x, blockCount.y, totalBlocks);
			break;
			   }
		}
		/*
		std::string errorstr = cudaGetErrorString(cudaPeekAtLastError());
		cout << errorstr << endl;;
		*/

		//checkCudaErrors(cudaThreadSynchronize());

		iterations++;

		// copy to pbo
		dim3 block(32, 32, 1);
		dim3 grid(ceil(imagesize.x / (float)block.x), ceil(imagesize.y / (float)block.y), 1);
		copy2pbo<<<grid,block>>>(cumulatedColor, pos, iterations, imagesize.x, imagesize.y, gamma);
		checkCudaErrors(cudaThreadSynchronize());
	}

	void launch_rendering_kernel_singlepass_mode(float3 *pos) {
		if( updateCamera ) {
			TrackBall& tball = window->trackball();
			mat4 mat(tball.getInverseMatrix());
			mat = mat.trans();

			vec3 camPos = cam.pos;
			vec3 camDir = cam.dir;
			vec3 camUp = cam.up;

			camPos = (mat * (camPos / tball.getScale()));
			camDir = (mat * camDir);
			camUp = (mat * camUp);

			Camera caminfo = cam;
			caminfo.dir = camDir.normalized();
			caminfo.up = camUp.normalized();
			caminfo.pos = camPos;
			caminfo.right = caminfo.dir.cross(caminfo.up).normalized();

			cudaMemcpy(d_cam, &caminfo, sizeof(Camera), cudaMemcpyHostToDevice);
		}

		//bindTexture2<<< 1, 1 >>>(d_texobjs, scene.getTextures().size());
		//setParams<<<1, 1>>>(specType, tracingType, scene.getEnvironmentMap());
		initScene<<<1,1024>>>(lights.size(), d_lights, shapes.size(), d_shapes, materials.size(), d_materials);

		dim3 block(32, 32, 1);
		dim3 grid(ceil(imagesize.x / (float)block.x), ceil(imagesize.y / (float)block.y), 1);

		int seed = iterations+rand();
		initPixel<<<grid, block>>>(
			thrust::raw_pointer_cast(&pixels[0]), thrust::raw_pointer_cast(&activePixels0[0]),
			d_cam, seed, imagesize.x, imagesize.y);
		//checkCudaErrors(cudaThreadSynchronize());

		PixelActivivtyTester tester(thrust::raw_pointer_cast(&pixels[0]));

		int count = activePixels0.size();
		const int maxBounces = scene.maxBounces();
		const int nthreads = 512;
		for(int i=0;i<maxBounces;i++) {
			// trace at most 65536 rays at a time
			const int maxRays = 65536;
			int nround = ceil(count / (float)maxRays);
			for(int j=0;j<nround;j++) {
				int startIdx = j * maxRays;
				int endIdx = min(startIdx + maxRays, count);
				raytrace_singlepass<<<ceil(maxRays/(float)nthreads), nthreads>>>( seed, i, thrust::raw_pointer_cast(&pixels[0]), thrust::raw_pointer_cast(&activePixels0[0]), startIdx, endIdx, endIdx-startIdx, imagesize.x, imagesize.y );
				checkCudaErrors(cudaThreadSynchronize());
			}

			thrust::device_vector<int>::iterator activeEnd = thrust::remove_if( activePixels0.begin(), activePixels0.begin()+count,  tester );
			count = activeEnd - activePixels0.begin();
			//cout << "bounce #" << i << " active count = " << count << endl;
#if 0
			const float PIXNUM_THRES = 0.01;
			if( count / (float)npixels() < PIXNUM_THRES ) break;	// break if only a few more active pixels
#else
			if( !count ) break;
#endif
		}

		collectPixelValue<<<grid, block>>>( cumulatedColor, thrust::raw_pointer_cast(&pixels[0]), imagesize.x, imagesize.y );

		//checkCudaErrors(cudaThreadSynchronize());

		iterations++;

		// copy to pbo
		copy2pbo<<<grid,block>>>(cumulatedColor, pos, iterations, imagesize.x, imagesize.y, gamma);
		checkCudaErrors(cudaThreadSynchronize());
	}

	void render() {
		sdkStartTimer(&timer);

		// map OpenGL buffer object for writing from CUDA
		float3 *dptr;
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
			cuda_vbo_resource));
		//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

		//launch_rendering_kernel(dptr, sMode);
		launch_rendering_kernel_singlepass_mode(dptr);

		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

		sdkStopTimer(&timer);
	}

	void run() {
		GLFWWindowManager::instance()->run();		
	}

	void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
	void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);
	int npixels() const { return imagesize.x * imagesize.y; }
	void loadScene(const string& filename);

	// scene information
	Camera cam;
	Scene scene;
	vector<Shape> shapes;
	vector<Material> materials;
	vector<int> lights;


	// device side resources
	Camera* d_cam;
	Shape* d_shapes;
	TextureObject* d_tex;
	cudaTextureObject_t* d_texobjs;
	int* d_lights;
	Material* d_materials;

	// rendering control
	int iterations;
	int sMode;
	float gamma;
	int AASamples, AASamples_old;
	int kernelIdx;
	int tracingType;
	int specType;
	bool updateCamera;

	// benchmarker
	StopWatchInterface *timer;
	int fpsCount;        // FPS count for averaging
	int fpsLimit;        // FPS limit for sampling
	int g_Index;
	float avgFPS;
	unsigned int frameCount;
	clock_t startTime, endTime;

	// rendered image
	int2 imagesize;
	float3* cumulatedColor;
	thrust::device_vector<PixelState> pixels;
	thrust::device_vector<int> activePixels0, activePixels1;


	// rendering related resources
	GLuint vbo;
	struct cudaGraphicsResource* cuda_vbo_resource;

	// result visualization
	RayTracerWindow* window;
};


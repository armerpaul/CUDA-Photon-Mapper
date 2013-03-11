/*
 * CPE 570 Final Project
 * Photon Mapper
 * Zoë Wood
 * Paul Armer (parmer)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "glm/glm.hpp"
#include <math.h>
#include <algorithm>
#include <assert.h>
//#include "Image.h"
#include "types.h"
#include "cudaPhotonMapper.h"
#include "kdtree.h"

// kdtree types from kd-tree code
// These are needed for kdtree functions to run on CUDA
struct kdnode {
	double *pos;
	int dir;
	void *data;

	struct kdnode *left, *right;
};

struct res_node {
	struct kdnode *item;
	double dist_sq;
	struct res_node *next;
};

struct kdtree {
	int dim;
	struct kdnode *root;
	struct kdhyperrect *rect;
	void (*destr)(void*);
};

struct kdres {
	struct kdtree *tree;
	struct res_node *rlist, *riter;
	int size;
};
 

Camera * camera, *cam_d;
RectLight *light, *l_d;
Plane * planes, *p_d;
Sphere * spheres, *s_d;
Photon * photonArray, *ph_d;
kdtree * kdTree;
struct kdres * kdresult;
int numPhotons, kdTreeIncomplete = 1;
float theta, stheta;

Camera* CameraInit();
PointLight* LightInit();
Sphere* CreateSpheres();
Plane* CreatePlanes();
__host__ __device__ Point CreatePoint(float x, float y, float z);
__host__ __device__ color_t CreateColor(float r, float g, float b);

__global__ void CUDAPhotonTrace(Plane * f, RectLight *l, Sphere * s, Photon * position);
__global__ void CUDARayTrace(Camera * cam, Plane * f, Sphere * s, kdtree * tree, uchar4 * pos);

__device__ Photon * kd_res_photonf(struct kdres *rset, float *pos);

__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, kdtree * tree);
__device__ Photon PhotonTrace(Photon p, Sphere* s, Plane* f);
__device__ color_t SphereShading(int sNdx, Ray r, Point p, Sphere* sphereList, PointLight* l);
__device__ color_t Shading(Ray r, Point p, Point normalVector, PointLight* l, color_t diffuse, color_t ambient, color_t specular); 
__device__ float SphereRayIntersection(Sphere* s, Ray r);
__device__ float PlaneRayIntersection(Plane* s, Ray r);

static void HandleError( cudaError_t err, const char * file, int line)
{
	if(err !=cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/* 
 *  Handles CUDA errors, taking from provided sample code on clupo site
 */
extern "C" void setup_scene()
{
	kdTree = kd_create(3);
	HANDLE_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	camera = CameraInit();
	light = new RectLight();
	spheres = CreateSpheres();
	planes = CreatePlanes(); 
	numPhotons = light->width * PHOTON_DENSITY * light->height * PHOTON_DENSITY;
	photonArray = (Photon *) malloc(sizeof(Photon) * numPhotons * NUM_BOUNCES); 
	
	HANDLE_ERROR( cudaMalloc((void**)&cam_d, sizeof(Camera)) );
	HANDLE_ERROR( cudaMalloc(&p_d, sizeof(Plane)*NUM_PLANES) );
	HANDLE_ERROR( cudaMalloc(&l_d, sizeof(RectLight)) );
	HANDLE_ERROR( cudaMalloc(&s_d,  sizeof(Sphere)*NUM_SPHERES));
	HANDLE_ERROR( cudaMalloc(&ph_d,  sizeof(Photon)*numPhotons*NUM_BOUNCES));

	HANDLE_ERROR( cudaMemcpy(l_d, light, sizeof(RectLight), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(cam_d, camera,sizeof(Camera), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(p_d, planes,sizeof(Plane)*NUM_PLANES, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
	
	theta = 0;
	stheta = 0;
}

/*
 * Function Wrapper for the kernel that shoots out photons
 */
extern "C" void photonLaunch()
{
	Point move;

	//light->position.x -= 2 *sin(theta += .01);	

	//spheres[NUM_SPHERES-1].radius=5;
	//spheres[NUM_SPHERES-1].center=light->position;
	//spheres[NUM_SPHERES-1].ambient=CreateColor(1,0,0);
	//spheres[NUM_SPHERES-1].diffuse=CreateColor(1,1,1);
	//spheres[NUM_SPHERES-1].specular=CreateColor(1,1,1);

	HANDLE_ERROR( cudaMemcpy(l_d, light, sizeof(RectLight), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(ph_d, photonArray, sizeof(Photon)*numPhotons*NUM_BOUNCES, cudaMemcpyHostToDevice) );

	// The Kernel Call
	dim3 gridSize((light->width * PHOTON_DENSITY + 15)/16, (light->height * PHOTON_DENSITY + 15)/16);
	dim3 blockSize(16,16);
	CUDAPhotonTrace<<< gridSize, blockSize  >>>(p_d, l_d, s_d, ph_d);
	cudaThreadSynchronize();

	 
	if (kdTreeIncomplete) {
		HANDLE_ERROR( cudaMemcpy(photonArray, ph_d, sizeof(Photon)*numPhotons, cudaMemcpyDeviceToHost) );

		kd_clear(kdTree);
		for(int i=0; i < numPhotons; i++) {
			assert(0 == kd_insert3(kdTree, photonArray[i].position.x, photonArray[i].position.y, photonArray[i].position.z, &photonArray[i]));
		
			//printf("%f %f %f to %f %f %f\n", photonArray[i].position.x, photonArray[i].position.y, photonArray[i].position.z,
			//		photonArray[i].direction.x, photonArray[i].direction.y, photonArray[i].direction.z);
		}
		kdTreeIncomplete = false;
	} else {
		// Might do something here...
	}
	
}

extern "C" void renderScene(uchar4 *pos)
{
	
	CUDARayTrace(cam_d, p_d, s_d, kdTree, pos);
	
	
}




/*
 * CUDA global function which performs ray tracing. Responsible for initializing and writing to output vector
 */
__global__ void CUDARayTrace(Camera * cam, Plane * f, Sphere * s, kdtree *tree, uchar4 * pos)
{
	float tanVal = tan(FOV/2);

	//CALCULATE ABSOLUTE ROW,COL
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	color_t returnColor;
	Ray r;

	//BOUNDARY CHECK
	if(row >= WINDOW_HEIGHT || col >= WINDOW_WIDTH)
		return;

	float rvaly = tanVal - (2 * tanVal / WINDOW_HEIGHT) * row;
	float rvalx = -1 * WINDOW_WIDTH / WINDOW_HEIGHT * tanVal + (2 * tanVal / WINDOW_HEIGHT) * col;
	//INIT RAY VALUES
	r.origin = cam->eye;
	r.direction = cam->lookAt;
	r.direction += (rvalx * cam->lookRight);
	r.direction += (rvaly * cam->lookUp);
	r.direction = glm::normalize(r.direction);
	//r.direction.y += tanVal - (2 * tanVal / WINDOW_HEIGHT) * row;
	//r.direction.x += -1 * WINDOW_WIDTH / WINDOW_HEIGHT * tanVal + (2 * tanVal / WINDOW_HEIGHT) * col;

	//RAY TRACE
	returnColor = RayTrace(r, s, f, tree);

	//CALC OUTPUT INDEX
	int index = row *WINDOW_WIDTH + col;

	//PLACE DATA IN INDEX
	pos[index].x = 0xFF * returnColor.r;
	pos[index].y = 0xFF * returnColor.g;
	pos[index].z = 0xFF * returnColor.b;
	pos[index].w = 0xFF * returnColor.f;

}

/*
 * Performs Ray tracing over all spheres for any ray r
 */
__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, kdtree * tree) {
	color_t color = CreateColor(0, 0, 0); 
	kdres * nearestPhotons;
	float t, smallest;
	int i = 0, closestSphere = -1, closestPlane = -1;
	//r.direction += r.origin; //Set back to normal
	Point normalVector;
	//FIND CLOSEST SPHERE ALONG RAY R
	while (i < NUM_SPHERES) {
		t = SphereRayIntersection(s + i, r);

		if (t > 0 && (closestSphere < 0 || t < smallest)) {
			smallest = t;
			closestSphere = i;
		}
		i++;
	}
	//r.direction -= r.origin;
	i=0;
	while (i < NUM_PLANES) {
		t = PlaneRayIntersection(f + i, r);
		if (t > 0 && ( (closestSphere < 0 && closestPlane < 0) || t < smallest)) {//POSSIBLE LOGIC FIX CLOSESTSPHERE >1
			smallest = t;
			closestSphere = -1;
			closestPlane = i;
		}
		i++;
	}

	//SETUP FOR SHADOW CALCULATIONS
	i = 0;
	Ray shadowRay;

	if(closestPlane > -1 || closestSphere > -1)
	{	
		float resPoint[3];
		float pos[] = {r.direction.x * smallest, 
				r.direction.y * smallest,
				 r.direction.z * smallest};
		nearestPhotons = kd_nearest_rangef(tree, pos, PHOTON_RANGE);
		float dist;
		Photon * data;
		while( !kd_res_end( nearestPhotons ) ) {
			data = kd_res_photonf( nearestPhotons, resPoint );
			dist = glm::distance(glm::vec3(pos[0], pos[1], pos[2]), glm::vec3(resPoint[0], resPoint[1], resPoint[2]));
			color.r += ((PHOTON_RANGE - dist) / PHOTON_RANGE) * data->color.r;
			color.g += ((PHOTON_RANGE - dist) / PHOTON_RANGE) * data->color.g;
			color.b += ((PHOTON_RANGE - dist) / PHOTON_RANGE) * data->color.b;
		}
	}
	
	return color;
}


/*
 * CUDA global function which performs photon mappning. Responsible for initializing and writing to output vector
 */
__global__ void CUDAPhotonTrace(Plane * f, RectLight * l, Sphere * s, Photon * pos)
{
	//CALCULATE ABSOLUTE ROW,COL
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	Photon ph = Photon();

	//BOUNDARY CHECK
	if(row >= l->height * PHOTON_DENSITY || col >= l->width * PHOTON_DENSITY)
		return;

	//INIT PHOTON VALUES - Doesn't support a moving light yet, makes assumptions about position
	ph.position = l->position;
	ph.position.x += (float)row / (float)(l->height);
	ph.position.z += (float)col / (float)(l->width);

	ph.direction.x = l->normal.x + (float)row / (float)l->height; //just in hopes that something interesting happens
	ph.direction.y = l->normal.y; // we'll just keep this one the same since rand ain't workin'
	ph.direction.z = l->normal.z + (float)col / (float)l->width;
	ph.direction = glm::normalize(ph.direction);

	//CALC OUTPUT INDEX
	int index = row * l->height * PHOTON_DENSITY + col;

	//PLACE PHOTON IN INDEX
	pos[index] = PhotonTrace(ph, s, f);

}



/*
 * Performs Ray tracing over all spheres for any ray r
 */
__device__ Photon PhotonTrace(Photon ph, Sphere* s, Plane* f) {
	
	color_t color = CreateColor(0, 0, 0); 
	float t, smallest;
	int i = 0, closestSphere = -1, closestPlane = -1;
	//r.direction += r.origin; //Set back to normal
	Point normalVector;
	Ray r;
	r.origin = ph.position;
	r.direction = ph.direction;
	//FIND CLOSEST SPHERE ALONG RAY R
	while (i < NUM_SPHERES) {
		t = SphereRayIntersection(s + i, r);

		if (t > 0 && (closestSphere < 0 || t < smallest)) {
			smallest = t;
			closestSphere = i;
		}
		i++;
	}
	//r.direction -= r.origin;
	i=0;
	while (i < NUM_PLANES) {
		t = PlaneRayIntersection(f + i, r);
		if (t > 0 && ( (closestSphere < 0 && closestPlane < 0) || t < smallest)) {//POSSIBLE LOGIC FIX CLOSESTSPHERE >1
			smallest = t;
			closestSphere = -1;
			closestPlane = i;
		}
		i++;
	}

	if (closestSphere > -1) {
		ph.position.x += smallest * ph.direction.x;
		ph.position.y += smallest * ph.direction.y;
		ph.position.z += smallest * ph.direction.z;

		ph.color = s[closestSphere].ambient;
	} else if (closestPlane > -1) {
		ph.position.x += smallest * ph.direction.x;
		ph.position.y += smallest * ph.direction.y;
		ph.position.z += smallest * ph.direction.z;	
		ph.color = f[closestPlane].ambient;
	}

	return ph;

}

/*
 * Determines distance of intersection of Ray with Plane, -1 returned if no intersection
 */
__device__ float PlaneRayIntersection(Plane *p, Ray r)
{
	float t;
	//Point N = glm::normalize(p->normal);
	float denominator = glm::dot(r.direction,p->normal);
	if(denominator!=0)
	{
		t = (glm::dot(p->center-r.origin,p->normal)) / denominator;
		if (t>1000000)
			return -1;
		return t;
	}
	else
	{
		return -1;
	}
}


/*
 * Determines distance of intersection of Ray with Sphere, -1 returned if no intersection
 * http://sci.tuomastonteri.fi/programming/sse/example3
 */
__device__ float SphereRayIntersection(Sphere* s, Ray r) {
	float a, b, c, d, t1, t2;

	a = glm::dot((r.direction), (r.direction));

	b = glm::dot((r.origin)-(s->center),(r.direction));
	c = glm::dot((s->center),(s->center)) +glm::dot(r.origin,r.origin) -2.0f*glm::dot(r.origin, s->center)
		- (s->radius * s->radius);
	d = (b * b) - (a * c);

	if (d >= 0) {

		t1 = (-1 * b - sqrt(d)) / (a);
		t2 = (-1 * b + sqrt(d)) / (a);

		if (t2 > t1 && t1 > 0) {
			return t1;

		} else if (t2 > 0) {
			return t2;
		}
	}
	return -1;
}


extern "C" void ijklMove(unsigned char key)
{
	float sin_theta_x, cos_theta_x, sin_theta_y,cos_theta_y;
	switch(key){
	case('i'):
		camera->theta_x+=.05;
		break; 
	case('k'):
		camera->theta_x-=.05;
		break;
	case('j'):
		camera->theta_y-=.05;
		break;
	case('l'):
		camera->theta_y+=.05;
		break;
	}
	sin_theta_x = sin(camera->theta_x);
	sin_theta_y = sin(camera->theta_y);
	cos_theta_x = cos(camera->theta_x);
	cos_theta_y = cos(camera->theta_y);

	camera->lookAt = glm::normalize(CreatePoint(sin_theta_y ,sin_theta_x , -1*cos_theta_x*cos_theta_y));
	camera->lookRight = glm::normalize(CreatePoint(cos_theta_y , 0 , sin_theta_y));
	camera->lookUp = glm::normalize(CreatePoint(0,cos_theta_x, sin_theta_x));
}



extern "C" void wasdMove(unsigned char key)
{
	Point move;
	switch(key){
	case('w'):
		move = 10.f * camera->lookAt;
		break; 
	case('s'):
		move = -10.f *camera->lookAt;
		break;
	case('a'):
		move = -10.f * camera->lookRight;
		break;
	case('d'):
		move = 10.f * camera->lookRight;
		break;
	}
	camera->eye += move;
}
extern "C" void misc(unsigned char key)
{
	Point center;
	switch(key){
	case('q'):
		{
			// just for testing - resets kdTree
			kdTreeIncomplete = 1;
			camera = CameraInit();
			break;
		}
	case('r'):
		{
			spheres = CreateSpheres();
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('-'):
		{
			for(int i = 0; i < NUM_SPHERES; i++)
				spheres[i].radius = glm::max(0.f, spheres[i].radius-1);
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('='):
		{
			for(int i = 0; i < NUM_SPHERES; i++)
				spheres[i].radius = glm::min(100.f, spheres[i].radius+1);
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('o'):
		{
			Point center = *new Point(0,0,-2400);
			center = *new Point(0,0,-2400);
			for(int i = 0; i < NUM_SPHERES; i++)
			{
				Point c_dir = glm::normalize(spheres[i].center - center);
				Point move_dir = glm::cross(c_dir, *new Point(0,1,0));
				spheres[i].center += 5.f*move_dir;
				spheres[i].center -= 5.f*c_dir;

			}
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('p'):
		{
			center = *new Point(0,0,-2400);
			for(int i = 0; i < NUM_SPHERES; i++)
			{
				Point c_dir = glm::normalize(spheres[i].center - center);
				Point move_dir = glm::cross(c_dir, *new Point(0,1,0));
				spheres[i].center -= 10.f*move_dir;
				spheres[i].center += 10.f*c_dir;

			}
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('['):
		{
			center = camera->eye;
			for(int i = 0; i < NUM_SPHERES; i++)
			{
				Point c_dir = glm::normalize(spheres[i].center - center);
				Point move_dir = glm::cross(c_dir, *new Point(0,1,0));
				spheres[i].center += 10.f*move_dir;
				//spheres[i].center -= 10.f*c_dir;

			}
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}

	case(']'):
		{
			center = camera->eye;
			for(int i = 0; i < NUM_SPHERES; i++)
			{
				Point c_dir = glm::normalize(spheres[i].center - center);
				Point move_dir = glm::cross(c_dir, *new Point(0,1,0));
				spheres[i].center -= 10.f*move_dir;
				//spheres[i].center += 10.f*c_dir;

			}
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('9'):
		{

			for(int i = 0; i < NUM_PLANES; i++) {
				planes[i].ambient.r = glm::max(planes[i].ambient.r - .05f, 0.f);
				planes[i].ambient.g = glm::max(planes[i].ambient.g - .05f, 0.f);
				planes[i].ambient.b = glm::max(planes[i].ambient.b - .05f, 0.f);
				planes[i].diffuse.r = glm::max(planes[i].diffuse.r - .05f, 0.f);
				planes[i].diffuse.g = glm::max(planes[i].diffuse.g - .05f, 0.f);
				planes[i].diffuse.b = glm::max(planes[i].diffuse.b - .05f, 0.f);
			}

			HANDLE_ERROR( cudaMemcpy(p_d, planes,sizeof(Plane)*NUM_PLANES, cudaMemcpyHostToDevice) );
			break;
		}
	case('0'):
		{
			for(int i = 0; i < NUM_PLANES; i++) {
				planes[i].ambient.r = glm::min(planes[i].ambient.r + .05f, 1.f);
				planes[i].ambient.g = glm::min(planes[i].ambient.g + .05f, 1.f);
				planes[i].ambient.b = glm::min(planes[i].ambient.b + .05f, 1.f);
				planes[i].diffuse.r = glm::min(planes[i].diffuse.r + .05f, 1.f);
				planes[i].diffuse.g = glm::min(planes[i].diffuse.g + .05f, 1.f);
				planes[i].diffuse.b = glm::min(planes[i].diffuse.b + .05f, 1.f);
			}

			HANDLE_ERROR( cudaMemcpy(p_d, planes,sizeof(Plane)*NUM_PLANES, cudaMemcpyHostToDevice) );
			break;
		}
	}
}


__device__ float fastSqrt( float number )
{
        long i;
        float x2, y;
        const float threehalfs = 1.5F;
 
        x2 = number * 0.5F;
        y  = number;
        i  = * ( long * ) &y;                       // evil floating point bit level hacking
        i  = 0x5f3759df - ( i >> 1 );               // what the actual fuck?
        y  = * ( float * ) &i;
        y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//      y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
 
        return 1 / y;
}

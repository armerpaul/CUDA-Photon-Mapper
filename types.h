#include "glm/glm.hpp"
#ifndef __TYPES_H__
#define __TYPES_H__

typedef glm::vec3 Point;

/* Color struct */
struct color_t {
	float r;
	float g;
	float b;
	float f; // "filter" or "alpha"
	
	color_t() :
   		r(0.f),
   		g(0.f),
   		b(0.f),
   		f(1.f)
   	{}
   	
	color_t(float r1, float b1, float g1) :
   		r(r1),
   		g(g1),
   		b(b1),
   		f(1.f)
   	{}
   	
   	color_t(float r1, float b1, float g1, float f1) :
   		r(r1),
   		g(g1),
   		b(b1),
   		f(f1)
   	{}
   	
};

typedef struct {
   Point center;
   float radius;
   color_t ambient, diffuse, specular;
} Sphere;

typedef struct {
   Point center, normal, topleft, bottomright;
   color_t ambient, diffuse, specular;
} Plane;

typedef struct {
   Point eye, lookAt, lookUp, lookRight;
  float theta_x, theta_y;
} Camera;

typedef struct {
   Point position;
   color_t ambient, diffuse, specular;
} PointLight;

typedef struct {
   Point origin, direction;
} Ray;

struct RectLight{
	int width;
	int height;
	Point position;
	Point normal;
	
	// The normal is 2 not 1 so that when we randomly generate photons, the math works :D
	RectLight() :
		width(10),
		height(10),
		position(Point(0, 100, -100)),
		normal(Point(0, -2, 0))
	{}
};

struct Photon {
	Point position;
	color_t color;
	Point direction;
	
	__host__ __device__ Photon() 
		: position(Point(0, 0, 0)),
		color(color_t()),
		direction(Point(0, 0, 0))
	{}
	
	__host__ __device__ Photon(Point p, color_t c, Point d)
		: position(p),
		color(c),
		direction(d)
	{}
};

struct PhotonData {
	color_t color;
	Point direction;
	
	PhotonData() :
		color(color_t()),
		direction(Point(0, 0, 0))
	{}
	
	PhotonData(color_t c, Point d) :
		color(c),
		direction(d)
	{}
};

#endif

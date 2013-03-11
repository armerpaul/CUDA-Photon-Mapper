/*
 * CPE 570 && CPE 458 Duet
 * Ray Tracer Header
 * Professor Christopher Lupo and Professor Zoe" Wood
 * Paul Armer(parmer), Bryan Ching(bcching), Matt Crussell(macrusse)
 */

#ifndef VANEXLIB_H
#define VANEXLIB_H

#ifdef _WIN32
#include <GL/glew.h>
#endif

#ifdef __unix__
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#endif

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024
#define SCREEN_DISTANCE -1
#define NUM_SPHERES 3
#define NUM_PLANES 6 
#define PHOTON_DENSITY 20
#define PHOTON_SPREAD 45 * 3.141592653589793 / 180

#define AMBIENT .2
#define DIFFUSE .6
#define SPECULAR .4

#define FOV 45 * 3.141592653589793 / 180

#endif

#pragma once
#include <math.h>
#include <stdio.h>

// This file contains various helper functions to implement Tomas Akenine-Mollers triangle-aabox test
// https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf

// Not ready for release, all these functions can be replaced by their GLM counterpart

#define X 0
#define Y 1
#define Z 2

#define CROSS(dest,v1,v2) \
dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
dest[2] = v1[0] * v2[1] - v1[1] * v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
dest[0] = v1[0] - v2[0]; \
dest[1] = v1[1] - v2[1]; \
dest[2] = v1[2] - v2[2];

#define FINDMINMAX(x0,x1,x2,min,max) \
min = max = x0;   \
if (x1 < min) min = x1; \
	if (x1 > max) max = x1; \
		if (x2 < min) min = x2; \
			if (x2 > max) max = x2;

__device__ int planeBoxOverlap(float normal[3], float vert[3], float maxbox[3])	// -NJMP-
{
	int q;
	float vmin[3], vmax[3], v;
	for (q = X; q <= Z; q++){
		v = vert[q];					// -NJMP-
		if (normal[q] > 0.0f){
			vmin[q] = -maxbox[q] - v;	// -NJMP-
			vmax[q] = maxbox[q] - v;	// -NJMP-
		}else{
			vmin[q] = maxbox[q] - v;	// -NJMP-
			vmax[q] = -maxbox[q] - v;	// -NJMP-
		}
	}
	if (DOT(normal, vmin) > 0.0f) return 0;	// -NJMP-
	if (DOT(normal, vmax) >= 0.0f) return 1;	// -NJMP-
	return 0;
}

/*======================== X-tests ========================*/

#define AXISTEST_X01(a, b, fa, fb)			   \
p0 = a * v0[Y] - b * v0[Z];			       	   \
p2 = a * v2[Y] - b * v2[Z];			       	   \
if (p0 < p2) { min = p0; max = p2; }\
else { min = p2; max = p0; } \
rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
if (min > rad || max < -rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)			   \
p0 = a * v0[Y] - b * v0[Z];			           \
p1 = a * v1[Y] - b * v1[Z];			       	   \
if (p0 < p1) { min = p0; max = p1; }\
else { min = p1; max = p0; } \
rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
if (min > rad || max < -rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)			   \
p0 = -a * v0[X] + b * v0[Z];		      	   \
p2 = -a * v2[X] + b * v2[Z];	       	       	   \
if (p0 < p2) { min = p0; max = p2; } \
else { min = p2; max = p0; } \
rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
if (min > rad || max < -rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)			   \
p0 = -a * v0[X] + b * v0[Z];		      	   \
p1 = -a * v1[X] + b * v1[Z];	     	       	   \
if (p0 < p1) { min = p0; max = p1; } \
else { min = p1; max = p0; } \
rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
if (min > rad || max < -rad) return 0;

/*======================== Z-tests ========================*/
#define AXISTEST_Z12(a, b, fa, fb)			   \
p1 = a * v1[X] - b * v1[Y];			           \
p2 = a * v2[X] - b * v2[Y];			       	   \
if (p2 < p1) { min = p2; max = p1; } \
else { min = p1; max = p2; } \
rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
if (min > rad || max < -rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)			   \
p0 = a * v0[X] - b * v0[Y];				   \
p1 = a * v1[X] - b * v1[Y];			           \
if (p0 < p1) { min = p0; max = p1; } \
else { min = p1; max = p0; } \
rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
if (min > rad || max < -rad) return 0;

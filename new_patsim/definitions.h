#include<stdlib.h>
#include<math.h>

#include "coordinate.h"
#include "physics.h"

#ifndef _definitions_h
#define _definitions_h

#define PI 3.141592653

// 移除硬编码的常量，改为外部变量
extern int MAX_NO_PARTICLES;  /* Maximum number of particles/processor */
extern int INIT_NO_PARTICLES; /* Initial number of particles/processor */
extern float MAX_INITIAL_VELOCITY;

extern float BOX_HORIZ_SIZE;
extern float BOX_VERT_SIZE;
#define WALL_LENGTH (2.0*BOX_HORIZ_SIZE+2.0*BOX_VERT_SIZE)

#define PARTICLE_BUFFER_SIZE MAX_NO_PARTICLES
#define COMM_BUFFER_SIZE  5*PARTICLE_BUFFER_SIZE

struct particle {
  pcord_t  pcord;
  int ptype;        /* Used to simulate mixing of gases */ 
};

typedef struct particle particle_t;

#endif

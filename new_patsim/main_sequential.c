#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "coordinate.h"
#include "definitions.h"
#include "physics.h"

// 定义全局变量
int MAX_NO_PARTICLES = 15000;
int INIT_NO_PARTICLES = 500;
float MAX_INITIAL_VELOCITY = 50.0;
float BOX_HORIZ_SIZE = 10000.0;
float BOX_VERT_SIZE = 10000.0;

// 测试模式下的预设值
#define TEST_PARTICLES 100
#define TEST_BOX_SIZE 100.0
#define TEST_VELOCITY 10.0

float rand1(){
	return (float)( rand()/(float) RAND_MAX );
}

void init_collisions(bool *collisions, unsigned int max){
	for(unsigned int i=0;i<max;++i)
		collisions[i]=0;
}

// 测试模式下的粒子初始化
void init_test_particles(pcord_t *particles) {
    // 预设的粒子位置和速度
    float test_positions[TEST_PARTICLES][2] = {
        {10.0, 10.0}, {20.0, 20.0}, {30.0, 30.0}, {40.0, 40.0}, {50.0, 50.0},
        {60.0, 60.0}, {70.0, 70.0}, {80.0, 80.0}, {90.0, 90.0}, {100.0, 100.0},
        {15.0, 15.0}, {25.0, 25.0}, {35.0, 35.0}, {45.0, 45.0}, {55.0, 55.0},
        {65.0, 65.0}, {75.0, 75.0}, {85.0, 85.0}, {95.0, 95.0}, {5.0, 5.0},
        {12.0, 12.0}, {22.0, 22.0}, {32.0, 32.0}, {42.0, 42.0}, {52.0, 52.0},
        {62.0, 62.0}, {72.0, 72.0}, {82.0, 82.0}, {92.0, 92.0}, {2.0, 2.0},
        {17.0, 17.0}, {27.0, 27.0}, {37.0, 37.0}, {47.0, 47.0}, {57.0, 57.0},
        {67.0, 67.0}, {77.0, 77.0}, {87.0, 87.0}, {97.0, 97.0}, {7.0, 7.0},
        {13.0, 13.0}, {23.0, 23.0}, {33.0, 33.0}, {43.0, 43.0}, {53.0, 53.0},
        {63.0, 63.0}, {73.0, 73.0}, {83.0, 83.0}, {93.0, 93.0}, {3.0, 3.0},
        {18.0, 18.0}, {28.0, 28.0}, {38.0, 38.0}, {48.0, 48.0}, {58.0, 58.0},
        {68.0, 68.0}, {78.0, 78.0}, {88.0, 88.0}, {98.0, 98.0}, {8.0, 8.0},
        {14.0, 14.0}, {24.0, 24.0}, {34.0, 34.0}, {44.0, 44.0}, {54.0, 54.0},
        {64.0, 64.0}, {74.0, 74.0}, {84.0, 84.0}, {94.0, 94.0}, {4.0, 4.0},
        {19.0, 19.0}, {29.0, 29.0}, {39.0, 39.0}, {49.0, 49.0}, {59.0, 59.0},
        {69.0, 69.0}, {79.0, 79.0}, {89.0, 89.0}, {99.0, 99.0}, {9.0, 9.0},
        {11.0, 11.0}, {21.0, 21.0}, {31.0, 31.0}, {41.0, 41.0}, {51.0, 51.0},
        {61.0, 61.0}, {71.0, 71.0}, {81.0, 81.0}, {91.0, 91.0}, {1.0, 1.0},
        {16.0, 16.0}, {26.0, 26.0}, {36.0, 36.0}, {46.0, 46.0}, {56.0, 56.0},
        {66.0, 66.0}, {76.0, 76.0}, {86.0, 86.0}, {96.0, 96.0}, {6.0, 6.0}
    };
    float test_velocities[TEST_PARTICLES][2] = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0},
        {6.0, 6.0}, {7.0, 7.0}, {8.0, 8.0}, {9.0, 9.0}, {10.0, 10.0},
        {1.5, 1.5}, {2.5, 2.5}, {3.5, 3.5}, {4.5, 4.5}, {5.5, 5.5},
        {6.5, 6.5}, {7.5, 7.5}, {8.5, 8.5}, {9.5, 9.5}, {0.5, 0.5},
        {1.2, 1.2}, {2.2, 2.2}, {3.2, 3.2}, {4.2, 4.2}, {5.2, 5.2},
        {6.2, 6.2}, {7.2, 7.2}, {8.2, 8.2}, {9.2, 9.2}, {0.2, 0.2},
        {1.7, 1.7}, {2.7, 2.7}, {3.7, 3.7}, {4.7, 4.7}, {5.7, 5.7},
        {6.7, 6.7}, {7.7, 7.7}, {8.7, 8.7}, {9.7, 9.7}, {0.7, 0.7},
        {1.3, 1.3}, {2.3, 2.3}, {3.3, 3.3}, {4.3, 4.3}, {5.3, 5.3},
        {6.3, 6.3}, {7.3, 7.3}, {8.3, 8.3}, {9.3, 9.3}, {0.3, 0.3},
        {1.8, 1.8}, {2.8, 2.8}, {3.8, 3.8}, {4.8, 4.8}, {5.8, 5.8},
        {6.8, 6.8}, {7.8, 7.8}, {8.8, 8.8}, {9.8, 9.8}, {0.8, 0.8},
        {1.4, 1.4}, {2.4, 2.4}, {3.4, 3.4}, {4.4, 4.4}, {5.4, 5.4},
        {6.4, 6.4}, {7.4, 7.4}, {8.4, 8.4}, {9.4, 9.4}, {0.4, 0.4},
        {1.9, 1.9}, {2.9, 2.9}, {3.9, 3.9}, {4.9, 4.9}, {5.9, 5.9},
        {6.9, 6.9}, {7.9, 7.9}, {8.9, 8.9}, {9.9, 9.9}, {0.9, 0.9},
        {1.1, 1.1}, {2.1, 2.1}, {3.1, 3.1}, {4.1, 4.1}, {5.1, 5.1},
        {6.1, 6.1}, {7.1, 7.1}, {8.1, 8.1}, {9.1, 9.1}, {0.1, 0.1},
        {1.6, 1.6}, {2.6, 2.6}, {3.6, 3.6}, {4.6, 4.6}, {5.6, 5.6},
        {6.6, 6.6}, {7.6, 7.6}, {8.6, 8.6}, {9.6, 9.6}, {0.6, 0.6}
    };

    for(int i = 0; i < TEST_PARTICLES; i++) {
        particles[i].x = test_positions[i][0];
        particles[i].y = test_positions[i][1];
        particles[i].vx = test_velocities[i][0];
        particles[i].vy = test_velocities[i][1];
    }
}

void print_usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s [options] simulation_time\n", prog_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help           Show this help message\n");
    fprintf(stderr, "  -t, --test           Run in test mode\n");
    fprintf(stderr, "  -p, --particles N    Set number of particles (default: 500)\n");
    fprintf(stderr, "  -b, --box-size N     Set box size (default: 10000)\n");
    fprintf(stderr, "  -v, --velocity N     Set max initial velocity (default: 50)\n");
    fprintf(stderr, "\nExample: %s -t 10\n", prog_name);
    exit(1);
}

int main(int argc, char** argv){
	unsigned int time_stamp = 0, time_max;
	float pressure = 0;
	bool test_mode = false;

	// 解析命令行参数
	for(int i = 1; i < argc; i++) {
		if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			print_usage(argv[0]);
		}
		else if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
			test_mode = true;
			INIT_NO_PARTICLES = TEST_PARTICLES;
			BOX_HORIZ_SIZE = BOX_VERT_SIZE = TEST_BOX_SIZE;
			MAX_INITIAL_VELOCITY = TEST_VELOCITY;
		}
		else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--particles") == 0) {
			if(++i < argc) INIT_NO_PARTICLES = atoi(argv[i]);
		}
		else if(strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--box-size") == 0) {
			if(++i < argc) BOX_HORIZ_SIZE = BOX_VERT_SIZE = atof(argv[i]);
		}
		else if(strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--velocity") == 0) {
			if(++i < argc) MAX_INITIAL_VELOCITY = atof(argv[i]);
		}
		else if(argv[i][0] != '-') {
			time_max = atoi(argv[i]);
		}
	}

	if(time_max == 0) {
		fprintf(stderr, "Error: simulation time is required\n");
		print_usage(argv[0]);
	}

	/* Initialize */
	// 1. set the walls
	cord_t wall;
	wall.y0 = wall.x0 = 0;
	wall.x1 = BOX_HORIZ_SIZE;
	wall.y1 = BOX_VERT_SIZE;

	// 2. allocate particle buffer and initialize the particles
	pcord_t *particles = (pcord_t*) malloc(INIT_NO_PARTICLES*sizeof(pcord_t));
	bool *collisions = (bool *)malloc(INIT_NO_PARTICLES*sizeof(bool));

	srand(time(NULL) + 1234);

	if(test_mode) {
		init_test_particles(particles);
	} else {
		float r, a;
		for(int i=0; i<INIT_NO_PARTICLES; i++){
			// initialize random position
			particles[i].x = wall.x0 + rand1()*BOX_HORIZ_SIZE;
			particles[i].y = wall.y0 + rand1()*BOX_VERT_SIZE;

			// initialize random velocity
			r = rand1()*MAX_INITIAL_VELOCITY;
			a = rand1()*2*PI;
			particles[i].vx = r*cos(a);
			particles[i].vy = r*sin(a);
		}
	}

	unsigned int p, pp;

	/* Main loop */
	for (time_stamp=0; time_stamp<time_max; time_stamp++) { // for each time stamp

		init_collisions(collisions, INIT_NO_PARTICLES);

		for(p=0; p<INIT_NO_PARTICLES; p++) { // for all particles
			if(collisions[p]) continue;

			/* check for collisions */
			for(pp=p+1; pp<INIT_NO_PARTICLES; pp++){
				if(collisions[pp]) continue;
				float t=collide(&particles[p], &particles[pp]);
				if(t!=-1){ // collision
					collisions[p]=collisions[pp]=1;
					interact(&particles[p], &particles[pp], t);
					break; // only check collision of two particles
				}
			}

		}

		// move particles that has not collided with another
		for(p=0; p<INIT_NO_PARTICLES; p++)
			if(!collisions[p]){
				feuler(&particles[p], 1);

				/* check for wall interaction and add the momentum */
				pressure += wall_collide(&particles[p], wall);
			}

	}

	printf("Average pressure = %f\n", pressure / (WALL_LENGTH*time_max));

	free(particles);
	free(collisions);

	return 0;
}


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>

#include "coordinate.h"
#include "definitions.h"
#include "physics.h"

float rand1() {
    return (float)(rand() / (float)RAND_MAX);
}

void init_collisions(bool *collisions, unsigned int max) {
    for (unsigned int i = 0; i < max; ++i)
        collisions[i] = 0;
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned int time_stamp = 0, time_max;
    float pressure = 0, local_pressure = 0;

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s simulation_time\n", argv[0]);
            exit(1);
        }
        MPI_Finalize();
        return 0;
    }

    time_max = atoi(argv[1]);

    cord_t wall = {0, 0, BOX_HORIZ_SIZE, BOX_VERT_SIZE};

    pcord_t *particles = (pcord_t*)malloc(INIT_NO_PARTICLES * sizeof(pcord_t));
    bool *collisions = (bool*)malloc(INIT_NO_PARTICLES * sizeof(bool));

    srand(time(NULL) + rank * 1234); // 各进程不同随机种子

    if (rank == 0) {
        float r, a;
        for (int i = 0; i < INIT_NO_PARTICLES; i++) {
            particles[i].x = wall.x0 + rand1() * BOX_HORIZ_SIZE;
            particles[i].y = wall.y0 + rand1() * BOX_VERT_SIZE;
            r = rand1() * MAX_INITIAL_VELOCITY;
            a = rand1() * 2 * PI;
            particles[i].vx = r * cos(a);
            particles[i].vy = r * sin(a);
        }
    }

    MPI_Bcast(particles, INIT_NO_PARTICLES * sizeof(pcord_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    int particles_per_proc = INIT_NO_PARTICLES / size;
    int start = rank * particles_per_proc;
    int end = (rank == size - 1) ? INIT_NO_PARTICLES : start + particles_per_proc;

    for (time_stamp = 0; time_stamp < time_max; time_stamp++) {
        init_collisions(collisions, INIT_NO_PARTICLES);

        // exam particles' collisions
        for (int p = start; p < end; p++) {
            if (collisions[p]) continue;
            for (int pp = p + 1; pp < INIT_NO_PARTICLES; pp++) {
                if (collisions[pp]) continue;
                float t = collide(&particles[p], &particles[pp]);
                if (t != -1) {
                    collisions[p] = collisions[pp] = true;
                    interact(&particles[p], &particles[pp], t);
                    break;
                }
            }
        }

        // renew particles position and record pressure
        local_pressure = 0;
        for (int p = start; p < end; p++) {
            if (!collisions[p]) {
                feuler(&particles[p], 1);
                local_pressure += wall_collide(&particles[p], wall);
            }
        }

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                      particles, particles_per_proc * sizeof(pcord_t), MPI_BYTE, MPI_COMM_WORLD);
    }

    MPI_Reduce(&local_pressure, &pressure, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Average pressure = %f\n", pressure / (WALL_LENGTH * time_max));
    }

    free(particles);
    free(collisions);
    MPI_Finalize();
    return 0;
}

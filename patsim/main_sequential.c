#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
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
    float local_pressure = 0, global_pressure = 0;

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s simulation_time\n", argv[0]);
            fprintf(stderr, "For example: %s 10\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }
    time_max = atoi(argv[1]);

    cord_t wall;
    wall.x0 = wall.y0 = 0;
    wall.x1 = BOX_HORIZ_SIZE;
    wall.y1 = BOX_VERT_SIZE;

    int grid_size = (int)sqrt(size);
    if (grid_size * grid_size != size) {
        if (rank == 0)
            fprintf(stderr, "Process count must be a perfect square.\n");// 进程数需要是平方数 才能构建size*size的二维网络
        MPI_Finalize();
        exit(1);
    }

    int px = rank % grid_size;
    int py = rank / grid_size;
    float region_width = BOX_HORIZ_SIZE / grid_size;
    float region_height = BOX_VERT_SIZE / grid_size;

    float x_start = px * region_width;
    float y_start = py * region_height;
    float x_end = x_start + region_width;
    float y_end = y_start + region_height;

    srand(time(NULL) + rank * 1234);

    int local_particle_count = 0;
    // 每个进程只保留自己区域内的粒子
    pcord_t *particles = (pcord_t*) malloc(INIT_NO_PARTICLES * sizeof(pcord_t));
    // 随机生成粒子位置和初速度
    for (int i = 0; i < INIT_NO_PARTICLES; i++) {
        float x = wall.x0 + rand1() * BOX_HORIZ_SIZE;
        float y = wall.y0 + rand1() * BOX_VERT_SIZE;
        if (x >= x_start && x < x_end && y >= y_start && y < y_end) {
            float r = rand1() * MAX_INITIAL_VELOCITY;
            float a = rand1() * 2 * PI;
            particles[local_particle_count].x = x;
            particles[local_particle_count].y = y;
            particles[local_particle_count].vx = r * cos(a);
            particles[local_particle_count].vy = r * sin(a);
            local_particle_count++;
        }
    }

    bool *collisions = (bool*)malloc(INIT_NO_PARTICLES * sizeof(bool));

    for (time_stamp = 0; time_stamp < time_max; time_stamp++) {
        init_collisions(collisions, local_particle_count); // 初始化碰撞状态

        // 检查粒子之间是否发生碰撞
        for (int p = 0; p < local_particle_count; p++) {
            if (collisions[p]) continue;
            for (int pp = p + 1; pp < local_particle_count; pp++) {
                if (collisions[pp]) continue;
                float t = collide(&particles[p], &particles[pp]);
                if (t != -1) {
                    collisions[p] = collisions[pp] = 1;
                    interact(&particles[p], &particles[pp], t);
                    break;
                }
            }
        }

        // 更新未碰撞粒子的位置
        for (int p = 0; p < local_particle_count; p++) {
            if (!collisions[p]) {
                feuler(&particles[p], 1);
                local_pressure += wall_collide(&particles[p], wall);
            }
        }

        // 粒子迁移逻辑（跨进程）
        pcord_t *send_buffers[4] = {NULL, NULL, NULL, NULL};
        int send_counts[4] = {0};

        for (int p = 0; p < local_particle_count; p++) {
            float x = particles[p].x;
            float y = particles[p].y;
            int dest_px = (int)(x / region_width);
            int dest_py = (int)(y / region_height);
            if (dest_px != px || dest_py != py) {
                int dest_rank = dest_py * grid_size + dest_px;
                int dir = (dest_py < py ? 0 : (dest_py > py ? 1 : (dest_px < px ? 2 : 3)));
                send_counts[dir]++;
                send_buffers[dir] = realloc(send_buffers[dir], send_counts[dir] * sizeof(pcord_t));
                send_buffers[dir][send_counts[dir]-1] = particles[p];
                particles[p].x = -1; // 标记为删除
            }
        }

        pcord_t *new_particles = NULL;
        int new_count = 0;
        for (int dir = 0; dir < 4; dir++) {
            int target_px = px + (dir == 2 ? -1 : (dir == 3 ? 1 : 0));
            int target_py = py + (dir == 0 ? -1 : (dir == 1 ? 1 : 0));
            if (target_px >= 0 && target_px < grid_size && target_py >= 0 && target_py < grid_size) {
                int dest_rank = target_py * grid_size + target_px;
                int send_count = send_counts[dir];
                int recv_count;
                // 先交换粒子数量
                MPI_Sendrecv(&send_count, 1, MPI_INT, dest_rank, 0,
                             &recv_count, 1, MPI_INT, dest_rank, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pcord_t *recv_buf = malloc(recv_count * sizeof(pcord_t));
                // 再交换粒子数据
                MPI_Sendrecv(send_buffers[dir], send_count * sizeof(pcord_t), MPI_BYTE, dest_rank, 1,
                             recv_buf, recv_count * sizeof(pcord_t), MPI_BYTE, dest_rank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                new_particles = realloc(new_particles, (new_count + recv_count) * sizeof(pcord_t));
                for (int i = 0; i < recv_count; i++) {
                    new_particles[new_count++] = recv_buf[i];
                }
                free(recv_buf);
            }
            free(send_buffers[dir]);
        }

        // 移除迁出粒子 添加迁入粒子 更新local_particle_count
        int i = 0;
        for (int p = 0; p < local_particle_count; p++) {
            if (particles[p].x != -1) {
                particles[i++] = particles[p];
            }
        }
        for (int j = 0; j < new_count; j++) {
            particles[i++] = new_particles[j];
        }
        local_particle_count = i;
        free(new_particles);
    }

    // 汇总压力
    MPI_Reduce(&local_pressure, &global_pressure, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Average pressure = %f\n", global_pressure / (WALL_LENGTH * time_max));
    }

    free(particles);
    free(collisions);
    MPI_Finalize();
    return 0;
}

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "coordinate.h"   //
#include "definitions.h"  //
#include "physics.h"      //

int MAX_NO_PARTICLES = 15000;
int INIT_NO_PARTICLES = 500;
float MAX_INITIAL_VELOCITY = 50.0;
float BOX_HORIZ_SIZE = 10000.0;
float BOX_VERT_SIZE = 10000.0;

#define TEST_PARTICLES 100
#define TEST_BOX_SIZE 100.0
#define TEST_VELOCITY 10.0

float rand1(){
	return (float)(rand()/(float)RAND_MAX);
}

void init_collisions_mpi(bool *coll_arr, unsigned int max_val){
	for(unsigned int k=0; k<max_val; ++k) coll_arr[k]=0;
}

void init_test_particles_mpi(pcord_t *particles, int rank, int P_size, float rank_y_min, float rank_y_max) {
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

    int particles_per_rank = TEST_PARTICLES / P_size;
    int start_idx = rank * particles_per_rank;
    int end_idx = (rank == P_size - 1) ? TEST_PARTICLES : (rank + 1) * particles_per_rank;

    for(int i = start_idx; i < end_idx; i++) {
        particles[i - start_idx].x = test_positions[i][0];
        particles[i - start_idx].y = test_positions[i][1];
        particles[i - start_idx].vx = test_velocities[i][0];
        particles[i - start_idx].vy = test_velocities[i][1];
    }
}

void print_usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s [options] simulation_time\n", prog_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help           Display help information\n");
    fprintf(stderr, "  -t, --test           Enable test mode\n");
    fprintf(stderr, "  -p, --particles N    Set number of particles per process (default: 10000)\n");
    fprintf(stderr, "  -b, --box-size N     Set box size (default: 10000)\n");
    fprintf(stderr, "  -v, --velocity N     Set maximum initial velocity (default: 50)\n");
    fprintf(stderr, "\nExample: mpirun -n 4 %s -t 10\n", prog_name);
    exit(1);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P_size);

    unsigned int time_max = 0;
    bool test_mode = false;
    unsigned int particles_per_rank_init = 10000; // 默认值

    // 解析命令行参数
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            if(rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
        else if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
            test_mode = true;
            particles_per_rank_init = TEST_PARTICLES / P_size;
            BOX_HORIZ_SIZE = BOX_VERT_SIZE = TEST_BOX_SIZE;
            MAX_INITIAL_VELOCITY = TEST_VELOCITY;
        }
        else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--particles") == 0) {
            if(++i < argc) particles_per_rank_init = atoi(argv[i]);
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
        if(rank == 0) {
            fprintf(stderr, "Error: simulation time is required\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // --- 区域分解（水平条带） ---
    float strip_actual_height = BOX_VERT_SIZE / P_size;
    float rank_y_min = rank * strip_actual_height;
    float rank_y_max = (rank + 1) * strip_actual_height;
    if (rank == P_size - 1) {
        rank_y_max = BOX_VERT_SIZE;
    }

    // --- 全局墙壁坐标 ---
    cord_t global_box_walls;
    global_box_walls.x0 = 0.0f;
    global_box_walls.y0 = 0.0f;
    global_box_walls.x1 = BOX_HORIZ_SIZE;
    global_box_walls.y1 = BOX_VERT_SIZE;

    // --- 粒子存储与初始化 ---
    pcord_t local_particle_array[MAX_NO_PARTICLES];
    bool particle_collision_flags[MAX_NO_PARTICLES];
    unsigned int current_local_particle_count = particles_per_rank_init;

    srand(time(NULL) + rank);

    if(test_mode) {
        init_test_particles_mpi(local_particle_array, rank, P_size, rank_y_min, rank_y_max);
    } else {
        for (unsigned int i = 0; i < current_local_particle_count; i++) {
            local_particle_array[i].x = rand1() * BOX_HORIZ_SIZE;
            float initial_y = rank_y_min + rand1() * (rank_y_max - rank_y_min);
            if (initial_y >= rank_y_max && rank != P_size -1) initial_y = rank_y_max - 1e-5f;
            if (initial_y < rank_y_min && rank != 0) initial_y = rank_y_min + 1e-5f;
            local_particle_array[i].y = initial_y;

            float r_vel = rand1() * MAX_INITIAL_VELOCITY;
            float angle_vel = rand1() * 2 * PI;
            local_particle_array[i].vx = r_vel * cos(angle_vel);
            local_particle_array[i].vy = r_vel * sin(angle_vel);
        }
    }

    // --- 粒子交换缓冲区 ---
    pcord_t send_buf_up[PARTICLE_BUFFER_SIZE];   //
    pcord_t send_buf_down[PARTICLE_BUFFER_SIZE]; //
    pcord_t recv_buf_up[PARTICLE_BUFFER_SIZE];   //
    pcord_t recv_buf_down[PARTICLE_BUFFER_SIZE]; //

    float rank_local_pressure = 0.0f;
    long particles_sent_up_stat = 0;
    long particles_sent_down_stat = 0;

    // --- 创建 pcord_t 的 MPI 数据类型 ---
    MPI_Datatype mpi_pcord_type;
    MPI_Type_contiguous(4, MPI_FLOAT, &mpi_pcord_type); // 4 个浮点数：x, y, vx, vy
    MPI_Type_commit(&mpi_pcord_type);

    // --- 主模拟循环 ---
    for (unsigned int t_stamp = 0; t_stamp < time_max; t_stamp++) {
        init_collisions_mpi(particle_collision_flags, current_local_particle_count);

        // 1. 局部成对碰撞
        for (unsigned int p_idx = 0; p_idx < current_local_particle_count; p_idx++) {
            if (particle_collision_flags[p_idx]) continue;
            for (unsigned int pp_idx = p_idx + 1; pp_idx < current_local_particle_count; pp_idx++) {
                if (particle_collision_flags[pp_idx]) continue;
                float time_to_collide = collide(&local_particle_array[p_idx], &local_particle_array[pp_idx]); //
                if (time_to_collide != -1.0f) {
                    interact(&local_particle_array[p_idx], &local_particle_array[pp_idx], time_to_collide); //
                    particle_collision_flags[p_idx] = true;
                    particle_collision_flags[pp_idx] = true;
                    break;
                }
            }
        }

        // 2. 移动未碰撞粒子与物理墙壁碰撞
        for (unsigned int p_idx = 0; p_idx < current_local_particle_count; p_idx++) {
            if (!particle_collision_flags[p_idx]) {
                feuler(&local_particle_array[p_idx], 1.0f); // STEP_SIZE 为 1.0
                rank_local_pressure += wall_collide(&local_particle_array[p_idx], global_box_walls); //
            }
        }

        // 3. 粒子迁移
        unsigned int num_to_send_up = 0;
        unsigned int num_to_send_down = 0;

        pcord_t temp_staying_particles[MAX_NO_PARTICLES];
        unsigned int count_staying_particles = 0;

        for (unsigned int p_idx = 0; p_idx < current_local_particle_count; p_idx++) {
            bool did_migrate = false;
            if (local_particle_array[p_idx].y < rank_y_min && rank > 0) { // 向上迁移到 rank-1
                if (num_to_send_up < PARTICLE_BUFFER_SIZE) {
                     send_buf_up[num_to_send_up++] = local_particle_array[p_idx];
                } // 否则：缓冲区溢出，处理错误或丢弃粒子
                did_migrate = true;
            } else if (local_particle_array[p_idx].y >= rank_y_max && rank < P_size - 1) { // 向下迁移到 rank+1
                if (num_to_send_down < PARTICLE_BUFFER_SIZE) {
                    send_buf_down[num_to_send_down++] = local_particle_array[p_idx];
                } // 否则：缓冲区溢出
                did_migrate = true;
            }

            if (!did_migrate) {
                if (count_staying_particles < MAX_NO_PARTICLES) {
                    temp_staying_particles[count_staying_particles++] = local_particle_array[p_idx];
                } // 否则：局部粒子数组溢出
            }
        }
        // 压缩 local_particle_array
        for(unsigned int i=0; i<count_staying_particles; ++i) local_particle_array[i] = temp_staying_particles[i];
        current_local_particle_count = count_staying_particles;

        particles_sent_up_stat += num_to_send_up;
        particles_sent_down_stat += num_to_send_down;

        // --- MPI 通信 ---
        unsigned int num_to_recv_up = 0;
        unsigned int num_to_recv_down = 0;
        MPI_Status status;

        // 与上方邻居 (rank-1) 交换
        if (rank > 0) { // 不是 rank 0
            MPI_Sendrecv(&num_to_send_up, 1, MPI_UNSIGNED, rank - 1, 0,
                         &num_to_recv_up, 1, MPI_UNSIGNED, rank - 1, 0,
                         MPI_COMM_WORLD, &status);
            MPI_Sendrecv(send_buf_up, num_to_send_up, mpi_pcord_type, rank - 1, 1,
                         recv_buf_up, num_to_recv_up, mpi_pcord_type, rank - 1, 1,
                         MPI_COMM_WORLD, &status);
        }
        // 与下方邻居 (rank+1) 交换
        if (rank < P_size - 1) { // 不是最后一个 rank
            MPI_Sendrecv(&num_to_send_down, 1, MPI_UNSIGNED, rank + 1, 0,
                         &num_to_recv_down, 1, MPI_UNSIGNED, rank + 1, 0,
                         MPI_COMM_WORLD, &status);
            MPI_Sendrecv(send_buf_down, num_to_send_down, mpi_pcord_type, rank + 1, 1,
                         recv_buf_down, num_to_recv_down, mpi_pcord_type, rank + 1, 1,
                         MPI_COMM_WORLD, &status);
        }

        // 添加接收到的粒子
        for (unsigned int i = 0; i < num_to_recv_up; i++) {
            if (current_local_particle_count < MAX_NO_PARTICLES) {
                local_particle_array[current_local_particle_count++] = recv_buf_up[i];
            } // 否则：溢出
        }
        for (unsigned int i = 0; i < num_to_recv_down; i++) {
            if (current_local_particle_count < MAX_NO_PARTICLES) {
                local_particle_array[current_local_particle_count++] = recv_buf_down[i];
            } // 否则：溢出
        }
        // MPI_Barrier(MPI_COMM_WORLD); // 可选：如果需要同步
    }

    // --- 聚合压力 ---
    float global_total_pressure = 0.0f;
    MPI_Reduce(&rank_local_pressure, &global_total_pressure, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // --- 聚合迁移统计（以 sent_up 为例） ---
    long global_total_sent_up = 0;
    MPI_Reduce(&particles_sent_up_stat, &global_total_sent_up, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    // 可以为 sent_down、received_up、received_down 添加类似的归约操作

    if (rank == 0) {
        float actual_circumference = 2.0f * (BOX_HORIZ_SIZE + BOX_VERT_SIZE);
        // global_total_pressure 是所有时间步和所有粒子中 wall_collide 记录的总动量。
        // 串行代码将此总和除以 (WALL_LENGTH * time_max)。
        // 这里应该相同。
        printf("Average pressure = %f\n", global_total_pressure / (actual_circumference * time_max));
        printf("Total number of particles migrated across all internal boundaries (sum of all ranks): %ld\n", global_total_sent_up);
        // 根据报告需要添加其他全局统计信息。
    }

    MPI_Type_free(&mpi_pcord_type);
    MPI_Finalize();
    return 0;
}
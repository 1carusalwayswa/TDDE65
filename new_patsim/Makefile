

sequential:main_sequential.c  physics.c
	gcc -std=c99 -g -o $@ $^ -lrt -lm

mpi:main_mpi.c  physics.c
	mpicc -std=c99 -g -o $@ $^ -lrt -lm

clean:
	-rm *.o sequential mpi

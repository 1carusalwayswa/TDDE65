CC = mpicc
CFLAGS = -std=c99 -g
LDFLAGS = -lrt -lm
TARGET = sequential
SRC = main_sequential.c physics.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

# sequential:main_sequential.c  physics.c
# 	mpicc -std=c99 -g -o $@ $^ -lrt -lm

# allclean:
# 	-rm *.o sequential
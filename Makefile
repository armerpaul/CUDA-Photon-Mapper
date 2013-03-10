CC=nvcc
LD=nvcc
CFLAGS= -c -O3 -Xlinker -framework,OpenGL,-framework,GLUT -DGL_GLEXT_PROTOTYPES 
LDFLAGS= -O3 -Xlinker -framework,OpenGL,-framework,GLUT -DGL_GLEXT_PROTOTYPES -lcudart -lm
CUDAFLAGS= -c -O3 -arch=sm_21

ALL= callbacksPBO.o cudaPhotonMapper.o simpleGLmain.o simplePBO.o kdtree.o

all= $(ALL) RT 

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o PhotonMapper

kdtree.o:	kdtree.c
	$(CC) $(CFLAGS) -o $@ $<

callbacksPBO.o:	callbacksPBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

kernelPBO.o:	kernelPBO.cu
	$(CC) $(CUDAFLAGS) -o $@ $<

cudaPhotonMapper.o:	cudaPhotonMapper.cu
	$(CC) $(CUDAFLAGS) -o $@ $< 


simpleGLmain.o:	simpleGLmain.cpp
	$(CC) $(CFLAGS) -o $@ $<

simplePBO.o: simplePBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*


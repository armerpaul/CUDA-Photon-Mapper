CC=nvcc
LD=nvcc
CFLAGS= -O3 -c -Xlinker -framework,OpenGL,-framework,GLUT -DGL_GLEXT_PROTOTYPES
LDFLAGS= -O3  -Xlinker -framework,OpenGL,-framework,GLUT -DGL_GLEXT_PROTOTYPES -lcudart
CUDAFLAGS= -O3 -c -arch=sm_21

ALL= callbacksPBO.o cudaRayTrace.o simpleGLmain.o simplePBO.o kdtree.o

all= $(ALL) Photonz

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o Photonz

kdtree.o:	kdtree.c
	$(CC) $(CFLAGS) -o $@ $<

callbacksPBO.o:	callbacksPBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

kernelPBO.o:	kernelPBO.cu
	$(CC) $(CUDAFLAGS) -o $@ $<

cudaRayTrace.o:	cudaRayTrace.cu
	$(CC) $(CUDAFLAGS) -o $@ $< 


simpleGLmain.o:	simpleGLmain.cpp
	$(CC) $(CFLAGS) -o $@ $<

simplePBO.o: simplePBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*


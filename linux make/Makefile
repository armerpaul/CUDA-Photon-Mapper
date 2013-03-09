CC=nvcc
LD=nvcc
CFLAGS= -O3 -c -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU 
LDFLAGS= -O3  -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU -lcudart kdtree-0.5.6/libkdtree.a -lm
CUDAFLAGS= -O3 -c -arch=sm_21

ALL= callbacksPBO.o cudaPhotonMapper.o simpleGLmain.o simplePBO.o

all= $(ALL) RT 

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o PhotonMapper

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


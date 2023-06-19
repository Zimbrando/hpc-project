EXE:=omp-sph cuda-sph
CFLAGS+=-std=c99 -Wall -Wpedantic
LDLIBS=-lm
BUILDDIR = ../build

.PHONY: clean

omp-sph: omp-sph.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $< $(LDLIBS) -fopenmp -o $(BUILDDIR)/$@

all: $(EXE)

cuda-sph: cuda-sph.cu
	@mkdir -p $(BUILDDIR)
	nvcc $< -o $(BUILDDIR)/$@

clean:
	\rm -f $(BUILDDIR)/*

# Makefile for the High Performance Computing programming project,
# Academic Year 2022/2023.
#
# Available targets:
#
# - sph: builds the non-GUI version (default)
#
# - sph.gui: builds the GUI version 
#
# - all: builds both the GUI and non-GUI versions
#
# - clean: clean up
#
# Last modified on 2022-11-27 by Moreno Marzolla

EXE:=omp-sph cuda-sph
CFLAGS+=-std=c99 -Wall -Wpedantic
LDLIBS=-lm
BUILDDIR = build

.PHONY: clean

omp-sph: sph.c
	$(CC) $<  $(LDLIBS) -o $(BUILDDIR)/$@

all: $(EXE)

cuda-sph: sph.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $(BUILDDIR)/$@

clean:
	\rm -f $(EXE) *.o *~

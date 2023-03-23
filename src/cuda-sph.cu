/****************************************************************************
 *
 * sph.c -- Smoothed Particle Hydrodynamics
 *
 * https://github.com/cerrno/mueller-sph
 *
 * Copyright (C) 2016 Lucas V. Schuermann
 * Copyright (C) 2022 Moreno Marzolla
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* "Particle-Based Fluid Simulation for Interactive Applications" by
   Müller et al. solver parameters */

__constant__ float Gx = 0.0, Gy = -10.0;   // external (gravitational) forces
__constant__ float REST_DENS = 300;    // rest density
__constant__ float GAS_CONST = 2000;   // const for equation of state
const __constant__ float H = 16;             // kernel radius
const __constant__ float EPS = 16;           // equal to H
__constant__ float MASS = 2.5;         // assume all particles have the same mass
__constant__ float VISC = 200;         // viscosity constant
__constant__ float DT = 0.0007;        // integration timestep
__constant__ float BOUND_DAMPING = -0.5;

#define WINDOW_WIDTH 3000
#define WINDOW_HEIGHT 2000

const int MAX_PARTICLES = 20000;
const int DAM_PARTICLES = 500;

const float VIEW_WIDTH = 1.5 * WINDOW_WIDTH;
const float VIEW_HEIGHT = 1.5 * WINDOW_HEIGHT;

/* Particle data structure; stores position, velocity, and force for
   integration stores density (rho) and pressure values for SPH.
*/
typedef struct {
    float *x, *y;         // position
    float *vx, *vy;       // velocity
    float *fx, *fy;       // force
    float *rho, *p;       // density, pressure
} particles_t;

particles_t *d_particles; // device structure
float *d_x, *d_y, *d_vx, *d_vy, *d_fx, *d_fy, *d_p, *d_rho; // device arrays inserted in the structure
int n_particles = 0;    // number of currently active particles

__global__ void kernel_main(particles_t *particles) {
    printf("\nrho: %f", particles->rho[0]);
    printf("\nx: %f, y: %f", particles->x[40], particles->y[60]);
}


/**
 * Return a random value in [a, b]
 */
float randab(float a, float b)
{
    return a + (b-a)*rand() / (float)(RAND_MAX);
}

/**
 * Set initial position of particle i to (x, y); initialize all
 * other attributes to default values (zeros).
 */
void init_particle(particles_t *particles, int i, float x, float y )
{
   particles->x[i] = x;
   particles->y[i] = y;
}

/**
 * Return nonzero iff (x, y) is within the frame
 */
int is_in_domain( float x, float y )
{
    return ((x < VIEW_WIDTH - EPS) &&
            (x > EPS) &&
            (y < VIEW_HEIGHT - EPS) &&
            (y > EPS));
}

/**
 * Initialize the SPH model with `n` particles. The caller is
 * responsible for allocating the `particles[]` array of size
 * `MAX_PARTICLES`.
 */
void init_sph( int n )
{
    n_particles = 0;
    printf("Initializing with %d particles\n", n);   
    const size_t array_size = sizeof(float) * n;
    
    /* Allocate in the device the arrays */
    cudaMalloc((void**)&d_x, array_size);
    cudaMalloc((void**)&d_y, array_size);
    cudaMalloc((void**)&d_vx, array_size);
    cudaMalloc((void**)&d_vy, array_size);
    cudaMalloc((void**)&d_fx, array_size);
    cudaMalloc((void**)&d_fy, array_size);
    cudaMalloc((void**)&d_p, array_size);
    cudaMalloc((void**)&d_rho, array_size);

    /* Create an host instance of the particles and initialize it */
    particles_t *particles = (particles_t*) malloc(sizeof(particles_t));
    particles->x = (float*) malloc(array_size);
    particles->y = (float*) malloc(array_size);
    particles->vx = (float*) calloc(n, sizeof(float));
    particles->vy = (float*) calloc(n, sizeof(float));
    particles->fx = (float*) calloc(n, sizeof(float));
    particles->fy = (float*) calloc(n, sizeof(float));
    particles->p = (float*) calloc(n, sizeof(float));
    particles->rho = (float*) calloc(n, sizeof(float));

    for (float y = EPS; y < VIEW_HEIGHT - EPS; y += H) {
        for (float x = EPS; x <= VIEW_WIDTH * 0.8f; x += H) {
            if (n_particles < n) {
                float jitter = rand() / (float)RAND_MAX;
                init_particle(particles, n_particles, x+jitter, y);
                n_particles++;
            }
        }
   }
   
   /* Copy the pointer to the device arrays in the device structure */
   cudaMalloc((void**)&d_particles, sizeof(particles_t));
   cudaMemcpy(&(d_particles->x), &d_x, sizeof(float*), cudaMemcpyHostToDevice); 
   cudaMemcpy(&(d_particles->y), &d_y, sizeof(float*), cudaMemcpyHostToDevice); 
   cudaMemcpy(&(d_particles->vx), &d_vx, sizeof(float*), cudaMemcpyHostToDevice); 
   cudaMemcpy(&(d_particles->vy), &d_vy, sizeof(float*), cudaMemcpyHostToDevice); 
   cudaMemcpy(&(d_particles->fx), &d_fx, sizeof(float*), cudaMemcpyHostToDevice); 
   cudaMemcpy(&(d_particles->fy), &d_fy, sizeof(float*), cudaMemcpyHostToDevice); 
   cudaMemcpy(&(d_particles->p), &d_p, sizeof(float*), cudaMemcpyHostToDevice); 
   cudaMemcpy(&(d_particles->rho), &d_rho, sizeof(float*), cudaMemcpyHostToDevice); 
   
   /* Copy host initialized particles data in the device arrays that are referenced in the device structure */
   cudaMemcpy(d_x, particles->x, array_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, particles->y, array_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_vx, particles->vx, array_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_vy, particles->vy, array_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_fx, particles->fx, array_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_fy, particles->fy, array_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_p, particles->p, array_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_rho, particles->rho, array_size, cudaMemcpyHostToDevice);
   
   assert(n_particles == n);    

   free(particles->x);
   free(particles->y);
   free(particles->vx);
   free(particles->vy);
   free(particles->fx);
   free(particles->fy);
   free(particles->p);
   free(particles->rho);
   free(particles);
}


int main(int argc, char **argv)
{
    srand(1234);

    int n = DAM_PARTICLES;
    int nsteps = 50;

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [nparticles [nsteps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (argc > 2) {
        nsteps = atoi(argv[2]);
    }

    if (n > MAX_PARTICLES) {
        fprintf(stderr, "FATAL: the maximum number of particles is %d\n", MAX_PARTICLES);
        return EXIT_FAILURE;
    }

    init_sph(n);
	
    kernel_main<<<1, 1>>>(d_particles);
    
    cudaDeviceSynchronize();
    //printf("Address: %x ", d_particles);
    /*
    double tstart, tstop;
    tstart = omp_get_wtime();
    for (int s=0; s<nsteps; s++) {
        update();
        const float avg = avg_velocities();
        if (s % 10 == 0)
            printf("step %5d, avgV=%f\n", s, avg);
    }
    tstop = omp_get_wtime();
    printf("Elapsed time: %fs ", tstop - tstart);
    free(particles);
    */
    
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_vx);
   cudaFree(d_vy);
   cudaFree(d_fx);
   cudaFree(d_fy);
   cudaFree(d_p);
   cudaFree(d_rho);
   cudaFree(d_particles);
   
   return EXIT_SUCCESS;
}

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

/* HPC - Assignment 2022/2023
   Marco Sternini - marco.sternini2@studio.unibo.it
   0000971418
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include "hpc.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* "Particle-Based Fluid Simulation for Interactive Applications" by
   Müller et al. solver parameters */

__constant__ float Gx = 0.0, Gy = -10.0;   // external (gravitational) forces
__constant__ float REST_DENS = 300;    // rest density
__constant__ float GAS_CONST = 2000;   // const for equation of state
__constant__ float MASS = 2.5;         // assume all particles have the same mass
__constant__ float VISC = 200;         // viscosity constant
__constant__ float DT = 0.0007;        // integration timestep
__constant__ float BOUND_DAMPING = -0.5;
__constant__ float POLY6;
__constant__ float SPIKY_GRAD;
__constant__ float VISC_LAP;
__constant__ float HSQ; 
const __constant__ float H = 16;             // kernel radius
const __constant__ float EPS = 16;           // equal to H

#define WINDOW_WIDTH 3000
#define WINDOW_HEIGHT 2000
#define BLKDIM 1024

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
float *d_avg_out; // device avg velocity reduction output
dim3 grid(0), block(0); // grid and block size
int n_particles = 0;    // number of currently active particles


__global__ void k_compute_density_pressure( particles_t *ps, int n )
{
    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const int i = threadIdx.x + blockIdx.x * blockDim.x;     

    if (i < n) {
        ps->rho[i] = 0.0;
        for (int j=0; j < n; j++) {
            const float dx = ps->x[j] - ps->x[i];
            const float dy = ps->y[j] - ps->y[i];
            const float d2 = dx*dx + dy*dy;

            if (d2 < HSQ) {
                ps->rho[i] += MASS * POLY6 * pow(HSQ - d2, 3.0);
            }
        }
        ps->p[i] = GAS_CONST * (ps->rho[i] - REST_DENS);
    }   
}

__global__ void k_compute_forces( particles_t *ps, int n )
{
    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const float EPS = 1e-6;

    if (i < n) {
        float fpress_x = 0.0, fpress_y = 0.0;
        float fvisc_x = 0.0, fvisc_y = 0.0;

        for (int j=0; j<n; j++) {
            if (i == j)
                continue;

            const float dx = ps->x[j] - ps->x[i];
            const float dy = ps->y[j] - ps->y[i];
            const float dist = hypotf(dx, dy) + EPS; // avoids division by zero later on

            if (dist < H) {
                const float norm_dx = dx / dist;
                const float norm_dy = dy / dist;
                // compute pressure force contribution
                fpress_x += -norm_dx * MASS * (ps->p[i] + ps->p[j]) / (2 * ps->rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                fpress_y += -norm_dy * MASS * (ps->p[i] + ps->p[j]) / (2 * ps->rho[j]) * SPIKY_GRAD * pow(H - dist, 3);
                // compute viscosity force contribution
                fvisc_x += VISC * MASS * (ps->vx[j] - ps->vx[i]) / ps->rho[j] * VISC_LAP * (H - dist);
                fvisc_y += VISC * MASS * (ps->vy[j] - ps->vy[i]) / ps->rho[j] * VISC_LAP * (H - dist);
            }
        }
        const float fgrav_x = Gx * MASS / ps->rho[i];
        const float fgrav_y = Gy * MASS / ps->rho[i];
        ps->fx[i] = fpress_x + fvisc_x + fgrav_x;
        ps->fy[i] = fpress_y + fvisc_y + fgrav_y;
    }
}

__global__ void k_integrate( particles_t *ps, int n )
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        // forward Euler integration
        ps->vx[i] += DT * ps->fx[i] / ps->rho[i];
        ps->vy[i] += DT * ps->fy[i] / ps->rho[i];
        ps->x[i] += DT * ps->vx[i];
        ps->y[i] += DT * ps->vy[i];

        // enforce boundary conditions
        if (ps->x[i] - EPS < 0.0) {
            ps->vx[i] *= BOUND_DAMPING;
            ps->x[i] = EPS;
        }
        if (ps->x[i] + EPS > VIEW_WIDTH) {
            ps->vx[i] *= BOUND_DAMPING;
            ps->x[i] = VIEW_WIDTH - EPS;
        }
        if (ps->y[i] - EPS < 0.0) {
            ps->vy[i] *= BOUND_DAMPING;
            ps->y[i] = EPS;
        }
        if (ps->y[i] + EPS > VIEW_HEIGHT) {
            ps->vy[i] *= BOUND_DAMPING;
            ps->y[i] = VIEW_HEIGHT - EPS;
        }
    }
}

/**
  * Calculates the avg velocity of the particles. The kernel operates a 
  * reduction. Each thread cooperates with threads of its block to obtain
  * a partial result that will be returned to the host. 
  * Unused threads fill the shared memory with zeroes to not interfere with
  * the result. 
  */
__global__ void k_avg_velocities( particles_t *ps, int n, float* o_result )
{
    __shared__ float s_results[BLKDIM];
    const int gind = threadIdx.x + blockIdx.x * blockDim.x;
    const int lind = threadIdx.x;
   
    /* If the thread is mapped to a non-existent
       particle put 0 */
    if (gind >= n ) {
        s_results[lind] = 0;
    } else {
        s_results[lind] = hypotf(ps->vx[gind], ps->vy[gind]) / n;
    }
    __syncthreads();

/* https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf Reduction#3 Sequential addressing */
    for (unsigned int s=blockDim.x/2; s>0; s/=2) {
        if (lind < s) {
            s_results[lind] += s_results[lind + s];
        }
        __syncthreads();
    }
    
    if (lind == 0) {
        o_result[blockIdx.x] = s_results[0];
    }
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
   particles->fx[i] = 0;
   particles->fy[i] = 0;
   particles->vx[i] = 0;
   particles->vy[i] = 0;
   particles->p[i] = 0;
   particles->rho[i] = 0;
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
 * Initialize the SPH model with `n` particles
 */
void init_sph( int n, int quiet )
{
    n_particles = 0;
    if (!quiet)
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

   /* Allocate avg velocity output arrays with the number of expected blocks */
   cudaMalloc((void**)&d_avg_out, sizeof(float) * ((n + BLKDIM-1)/BLKDIM));

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

/**
  * Initialize device __constant__ memory
  */ 
void init_constants( void ) 
{
    size_t size = sizeof(float);
    float H_HSQ = H * H; 
    float H_POLY6 = 4.0 / (M_PI * pow(H, 8));
    float H_SPIKY_GRAD = -10.0 / (M_PI * pow(H, 5));
    float H_VISC_LAP = 40.0 / (M_PI * pow(H, 5));

    cudaMemcpyToSymbol(HSQ, &H_HSQ, size);
    cudaMemcpyToSymbol(POLY6, &H_POLY6, size);
    cudaMemcpyToSymbol(SPIKY_GRAD, &H_SPIKY_GRAD, size);
    cudaMemcpyToSymbol(VISC_LAP, &H_VISC_LAP, size);
}

void update( void )
{
    /* Launch all the kernels in queue */
    k_compute_density_pressure<<<grid, block>>>(d_particles, n_particles);
    k_compute_forces<<<grid, block>>>(d_particles, n_particles);
    k_integrate<<<grid, block>>>(d_particles, n_particles);
}


int main(int argc, char **argv)
{
    srand(1234);

    int n = DAM_PARTICLES;
    int nsteps = 50;
    int opt;
    int quiet = 0;

    while ((opt = getopt(argc, argv, "p:s:q")) != -1) {
        switch (opt) {
            case 'p':
                n = atoi(optarg);
                break;
            case 's':
                nsteps = atoi(optarg);
                break;
            case 'q':
                quiet = 1;
                break;
            default:
                fprintf(stderr, "Usage: %s [-p number of particles] [-s number of steps] [-q]\n", argv[0]);
                return EXIT_FAILURE;
        }
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
    
    init_sph(n, quiet);
    init_constants();
    block = dim3(BLKDIM);
    grid = dim3((n_particles+BLKDIM-1)/BLKDIM);
    float *partial_avg_out = (float*)malloc(sizeof(float) * grid.x), avg_velocity = 0; // device reduction output
    double tstart, tstop;
    tstart = hpc_gettime();
    for (int s=0; s<nsteps; s++) {
        update();
	// Calculate avg velocity
	k_avg_velocities<<<grid, block>>>(d_particles, n_particles, d_avg_out);
	/* Computes the avg velocity each iteration to 
	   mantain an equal amount of workload */
	avg_velocity = 0;
	cudaMemcpy(partial_avg_out, d_avg_out, sizeof(float) * grid.x, cudaMemcpyDeviceToHost);
        for (int i=0; i < grid.x; i++) {
	    /* The host uses the partial results of each block and 
	       calculates the avg_velocity */
	    avg_velocity += partial_avg_out[i];
	}

	if (s % 10 == 0 && !quiet)
            printf("step %5d, avgV=%f\n", s, avg_velocity);
    }
    cudaDeviceSynchronize();
    tstop = hpc_gettime();
    printf("Execution time %fs ", tstop - tstart);
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_p);
    cudaFree(d_rho);
    cudaFree(d_particles);
    cudaFree(d_avg_out);
    free(partial_avg_out);
    return EXIT_SUCCESS;
}

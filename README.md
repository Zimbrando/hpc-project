# hpc-project
Assignment for the high performance computing course at @unibo

# Build
Compile it using make in the **src/** directory

Omp:
```
  make omp-sph
```
Cuda:
```
  make cuda-sph
```
The executable will be placed inside the **build/** directory

# Run

Both versions work with the same parameters
```
  ./omp-sph [Num particles] [Num simulation steps]
```

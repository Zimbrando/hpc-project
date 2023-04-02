# Marco Sternini - marco.sternini2@studio.unibo.it
# 0000971418
# OMP parallel - Benchmark measuring speedup, strong and weak scaling efficiency 

PROG=../build/omp-sph

if [ ! -f "$PROG" ]; then
    echo
    echo "Program $PROG not found, run 'make omp-sph' to build it"
    echo
    exit 1	
fi

#Measure speedup

SIZE=5000
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

for p in `seq $CORES`; do
    echo -n -e "$p\t"
    SUM=0
    for rep in `seq 5`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" -p$SIZE -q | sed 's/Execution time //' )"
        NUM=${EXEC_TIME:0:-2}
        SUM=$(echo "$SUM + $NUM" | bc -l)
        echo -n -e "${EXEC_TIME}\t"
    done
    AVG=$(echo "scale=4; $SUM / 5" | bc -l)
    echo "AVG $AVG s"
done

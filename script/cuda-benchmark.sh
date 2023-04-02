# Marco Sternini - marco.sternini2@studio.unibo.it
# 0000971418
# CUDA - Benchmark measuring throughput 

PROG=../build/cuda-sph

if [ ! -f "$PROG" ]; then
	echo
	echo "Program $PROG not found, run 'make cuda-sph' to build it"
	echo
	exit 1	
fi


for st in {50..200..50}; do
	echo "$st steps"
	for ps in {5000..20000..1000}; do
		echo -n -e "$ps\t"
		SUM=0
		for rep in `seq 5`; do
			EXEC_TIME="$( "$PROG" -p$ps -s$st -q | sed 's/Execution time //' )"
			NUM=${EXEC_TIME:0:-2}
			SUM=$(echo "$SUM + $NUM" | bc -l)
			echo -n -e "${EXEC_TIME}\t"
		done
		AVG=$(echo "scale=4; $SUM / 5" | bc -l)
		TH=$(echo "scale=4; ($st * $ps) / $AVG" | bc -l)
		echo "AVG $AVG s; TH $TH p/s"
	done
done
